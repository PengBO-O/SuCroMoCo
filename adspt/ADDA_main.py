import pathlib
import sys

import time
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
import pandas as pd

from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator

from ADDA_model import SoftPromptLM, Classifier, Discriminator
from ADDA_FT import finetuning, joint_finetuning
from ADDA_adaptation import adapting

sys.path.append("..")
from utils import MyDataset, encoding, load_data


def parser_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--gen_lr', type=float, default=5e-5,
                        help='learning rate')

    parser.add_argument('--dis_lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--adapt_epoch', type=int, default=5)
    parser.add_argument('--ft_epoch', type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--percent', type=float, default=0.1, help='sampling percentage of each category')

    parser.add_argument('--base_model', type=str, default='roberta-base',
                        choices=['bert-base-uncased', 'roberta-base'])
    parser.add_argument('--pooler', type=str, default='mean', choices=['cls', 'mean', 'pooled'])
    parser.add_argument('--dataset', type=str, default='stocksen', choices=['stocksen', 'tweetfinsent'])

    parser.add_argument("--sample", default=False, type=lambda x: (str(x).lower() in ['true', 1, 'yes']))

    _args_ = parser.parse_args()

    return _args_


def preparation():
    global args
    repo_root = pathlib.Path(__file__).parent.parent
    args.data_path = repo_root / 'data'
    args.time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    args.checkpoint = repo_root.joinpath(f'./save/ADDA/soft_prompt/{args.dataset}/{args.base_model}/')
    args.checkpoint.mkdir(parents=True, exist_ok=True)

    logger.add(args.checkpoint.joinpath(f'{args.time}_log.txt'))

    logger.info('ADDA_main.py')
    logger.info(f'backbone: {args.base_model}')
    logger.info(f"pooling type: {args.pooler}")
    logger.info(f'adapt epochs: {args.adapt_epoch}')
    logger.info(f'ft epochs: {args.ft_epoch}')
    logger.info(f'batch size: {args.batch_size}')
    logger.info(f'FT Learning rate: {args.lr:.3e}')
    logger.info(f'Gen Learning rate: {args.gen_lr:.3e}')
    logger.info(f'Dis Learning rate: {args.dis_lr:.3e}')

    logger.info(f'Dataset: {args.dataset}')
    if args.dataset == 'StockSen':
        args.class_num = 2
        args.print_step = 100
    elif args.dataset == 'TweetFinSent':
        args.class_num = 3
        args.print_step = 10
    logger.info(f"Category number: {args.class_num}")


def loader_building(data, tokenizer, bs: int, do_shuffle=False):
    encodings = encoding(data, tokenizer=tokenizer)
    loader = DataLoader(MyDataset(encodings),
                        batch_size=bs, shuffle=do_shuffle)
    return loader


def main():
    global args
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    if args.class_num == 2:
        src_data = load_data(f"{args.data_path}/sst2_train.csv")
        src_eval_data = load_data(f"{args.data_path}/sst2_test.csv")
    else:
        src_item = load_data(f"{args.data_path}/fpb_allagree.csv")
        src_df = pd.DataFrame.from_dict(src_item)
        src_data_df, src_eval_data_df = train_test_split(src_df, test_size=0.2, random_state=42)
        src_data, src_eval_data = src_data_df.to_dict(orient='list'), src_eval_data_df.to_dict(orient='list')

    if args.dataset in ['stocksen', 'tweetfinsent']:

        tgt_data = load_data(f"{args.data_path}/{args.dataset}_train.csv")
        tgt_eval_data = load_data(f"{args.data_path}/{args.dataset}_test.csv")

    elif args.dataset in ['semevel', 'fpb']:

        tgt_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_train_{args.k}.csv")
        tgt_eval_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_test_{args.k}.csv")

    else:
        raise ValueError("No such dataset")

    logger.info(f"{len(src_data)} samples from source")
    logger.info(f"{len(tgt_data)} samples from target")

    src_per = len(src_data) / (len(src_data) + len(tgt_data))
    tgt_per = 1 - src_per

    src_loader = loader_building(src_data, tokenizer=tokenizer,
                                 bs=int(round(src_per * args.batch_size)),
                                 do_shuffle=True)

    src_eval_loader = loader_building(src_eval_data, tokenizer=tokenizer,
                                      bs=args.batch_size,
                                      do_shuffle=True)

    tgt_loader = loader_building(tgt_data, tokenizer=tokenizer,
                                 bs=int(round(tgt_per * args.batch_size)),
                                 do_shuffle=True)
    tgt_eval_loader = loader_building(tgt_eval_data, tokenizer=tokenizer,
                                      bs=args.batch_size,
                                      do_shuffle=False)

    logger.info(f"source dataloader length {len(src_loader)}")
    logger.info(f"target dataloader length {len(tgt_loader)}")

    base_model = AutoModel.from_pretrained(args.base_model)
    src_encoder = SoftPromptLM(base_model,
                               pooling=args.pooler)
    tgt_encoder = SoftPromptLM(base_model,
                               pooling=args.pooler)

    classifier = Classifier(num_class=args.class_num)

    discriminator = Discriminator()

    (src_loader, src_eval_loader,
     tgt_loader, tgt_eval_loader,
     src_encoder, tgt_encoder,
     classifier, discriminator) = accelerator.prepare(src_loader, src_eval_loader,
                                                      tgt_loader, tgt_eval_loader,
                                                      src_encoder, tgt_encoder,
                                                      classifier, discriminator)

    logger.info("=== Training source encoder and classifier on source domain ===")
    src_encoder, classifier = finetuning(src_encoder, classifier,
                                         src_loader, src_eval_loader, accelerator,
                                         args, logger, domain='source', model_return=True)

    logger.info("=== Adversarial Adaptation ===")
    tgt_encoder = adapting(src_encoder, tgt_encoder, discriminator, src_loader, tgt_loader, accelerator, args, logger)

    logger.info("=== Joint Fine-tuning and evaluating on target domain ===")
    # finetuning(tgt_encoder, classifier,
    #            tgt_loader, tgt_eval_loader, accelerator,
    #            args, logger, domain='target', model_return=False, recording=True)
    joint_finetuning(src_encoder, tgt_encoder, classifier,
                     src_loader, tgt_loader, tgt_eval_loader,
                     accelerator, args, logger)
    #
    # logger.info("=== Evaluate on target domain ===")
    # results = evaluating(tgt_encoder, classifier, tgt_eval_loader)
    # logger.info(f"Acc: {results['accuracy']}, F1:{results['f1']}")

    return


if __name__ == '__main__':
    args = parser_args()
    preparation()
    main()
