import pathlib
import time
import sys
import argparse
from loguru import logger
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel, get_scheduler
from accelerate import Accelerator

from dual_loss import DualLoss

sys.path.append("..")
from utils import MyDataset, encoding, EarlyStop, load_data
from models import DualLM


def parser_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--percent', type=float, default=0.03, help='sampling percentage of each category')
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument('--base_model', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'roberta-base'])
    parser.add_argument('--dataset', type=str, default='TweetFinSent', choices=['StockSen', 'TweetFinSent', 'SST2'])

    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5, help='balance weight')

    parser.add_argument("--sample", default=False, type=lambda x: (str(x).lower() in ['true', 1, 'yes']))

    _args_ = parser.parse_args()

    return _args_


def preparation():
    global args
    repo_root = pathlib.Path(__file__).parent.parent
    args.data_path = repo_root / 'data'
    args.time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    args.checkpoint = repo_root.joinpath(f'/save/dual_contrast/joint/{args.dataset}/{args.base_model}/temp_{args.temperature}/')
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    logger.add(args.checkpoint.joinpath(f'{args.time}_log.txt'))

    logger.info('dualcontrast.py')
    logger.info(f'backbone: {args.base_model}')
    logger.info(f'Epochs: {args.epoch}')
    logger.info(f'total BS: {args.batch_size}')
    logger.info(f'Learning rate: {args.lr:.3e}')
    logger.info(f'Temperature: {args.temperature}')

    logger.info(f'Dataset: {args.dataset}')
    if args.dataset == 'stocksen':
        args.class_num = 2
    else:
        args.class_num = 3
    args.print_step = 50

def add_label_words(data, label2word):
    label_words = ' '.join(label2word.values())
    for i in range(len(data['text'])):
        data['text'][i] = label_words + ' ' + data['text'][i]
    return data


def evaluating(model, eval_loader):
    labels, predictions = [], []
    for _, batch in enumerate(eval_loader):
        eval_inputs = {k: batch[k] for k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            eval_outputs = model(**eval_inputs)
        preds = eval_outputs['predicts'].argmax(dim=-1)
        predictions.extend(preds.detach().cpu().tolist())
        labels.extend(batch['labels'].detach().cpu().tolist())

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return acc, f1


def train_fct(model, train_loader, eval_loader):
    global args
    earlystop = EarlyStop(patience=args.patience)
    training_steps = len(train_loader) * args.epoch
    accelerator = Accelerator()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=training_steps * 0.1,
        num_training_steps=training_steps
    )
    progress_bar = tqdm(range(training_steps))
    criterion = DualLoss(alpha=args.alpha, temp=args.temperature)
    train_loader, eval_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, eval_loader,
                                                                                 model,
                                                                                 optimizer,
                                                                                 scheduler)

    acc_best, f1_best = 0, 0
    total_time =0.0
    for epoch in range(args.epoch):
        epoch_start_time = time.time()

        model.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: batch[k] for k in ['input_ids', 'attention_mask']}
            outputs = model(**inputs)
            loss = criterion(outputs, batch['labels'])
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if (step + 1) % args.print_step == 0:
                _lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(f"learning rate: {_lr:.3e}")
                logger.info(f'loss: {(loss.item()):.5f}')

        epoch_end_time = time.time()
        total_time += epoch_end_time - epoch_start_time
        model.eval()
        acc, macro_f1 = evaluating(model, eval_loader)
        logger.info(f" acc: {acc * 100:.3f}, macro f1: {macro_f1 * 100:.3f}")

        logger.info('=' * 30)
        logger.info(f"End of Epoch {epoch}")

        if f1_best < macro_f1:
            f1_best = macro_f1
            acc_best = acc

        earlystop.update(macro_f1)
        if earlystop.stop:
            logger.info(f" Average training time of an epoch: {total_time / (epoch + 1)}")

            break

    with open(args.checkpoint.joinpath('/result.txt'), 'a', encoding='utf-8') as wf:
        wf.write(f"{acc_best * 100:.3f}\t{f1_best * 100:.3f}\n")

    logger.info(f" Best acc: {acc_best * 100:.3f}, macro f1: {f1_best * 100:.3f}")
    logger.info('Classification training finished')
    return


def main():
    global args
    if args.class_num == 2:
        pro_data = load_data(f"{args.data_path}/sst2_train.csv")
    else:
        pro_data = load_data(f"{args.data_path}/fpb_allagree.csv")

    if args.dataset in ['stocksen', 'tweetfinsent']:
        fin_data = load_data(f"{args.data_path}/{args.dataset}_train.csv")
        fin_eval_data = load_data(f"{args.data_path}/{args.dataset}_test.csv")
        label2word = {0: 'negative', 1: 'positive'}
    elif args.dataset in ['semevel', 'fpb']:
        fin_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_train_{args.k}.csv")
        fin_eval_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_test_{args.k}.csv")
        label2word = {0: 'negative', 1: 'neutral', 2: 'positive'}
    else:
        raise ValueError(f"No such dataset")

    raw_train_data = {}
    for k in fin_data.keys():
        raw_train_data[k] = fin_data[k] + pro_data[k]

    eval_data = add_label_words(fin_eval_data, label2word)
    train_data = add_label_words(raw_train_data, label2word)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    train_encodings = encoding(train_data, tokenizer=tokenizer)
    train_loader = DataLoader(MyDataset(train_encodings), batch_size=args.batch_size, shuffle=True)

    eval_encodings = encoding(eval_data, tokenizer=tokenizer)
    eval_loader = DataLoader(MyDataset(eval_encodings), batch_size=args.batch_size, shuffle=False)

    base_model = AutoModel.from_pretrained(args.base_model)
    model = DualLM(base_model=base_model, num_classes=args.class_num)
    train_fct(model, train_loader, eval_loader)


if __name__ == '__main__':
    args = parser_args()
    preparation()
    main()
