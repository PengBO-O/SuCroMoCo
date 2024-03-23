import os
import time
import random
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import get_scheduler, DataCollatorWithPadding


class MyDataset(Dataset):
    def __init__(self, item):
        super().__init__()

        self.item = item

    def __len__(self):
        return len(self.item['labels'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.item.items()}


def encoding(items, tokenizer):
    encodings = tokenizer(items['text'], padding=True, truncation=True,
                          add_special_tokens=True, return_token_type_ids=False,
                          return_tensors='pt')
    encodings['labels'] = torch.tensor(items['labels'], dtype=torch.long)

    return encodings


def load_data(file):
    df = pd.read_csv(file, sep='\t', encoding='utf-8', header=0, index_col=False)
    return df.to_dict(orient='list')


def loader_process(args, tokenizer, dataset, accelerator):
    def process_fn(examples):
        texts = examples['text']
        result = tokenizer(texts, max_length=args.max_length, truncation=True)
        result["labels"] = examples["labels"]
        return result

    tokenized_dataset = dataset.map(process_fn, batched=True, remove_columns=['text'])
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    dataloader = DataLoader(tokenized_dataset, shuffle=True,
                            collate_fn=data_collator, batch_size=args.batch_size)
    return dataloader


class EarlyStop:
    def __init__(self, patience):
        self.count = 0
        self.max_score = 0
        self.patience = patience
        self.stop = False

    def update(self, score):
        if score < self.max_score:
            self.count += 1
        else:
            self.count = 0
            self.max_score = score

        if self.count > self.patience:
            self.stop = True


def eval_fn(model, eval_loader):
    labels, predictions = [], []
    model.eval()
    for _, batch in enumerate(eval_loader):
        model_outputs = model(**batch)
        preds = model_outputs['logits'].argmax(dim=-1)
        predictions.extend(preds.detach().cpu().tolist())
        labels.extend(batch['labels'].detach().cpu().tolist())

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    results = {'accuracy': acc * 100, 'f1': f1 * 100}
    return results


def train_fn(args, train_loader, eval_loader, model, accelerator, logger):
    earlystop = EarlyStop(patience=args.patience)
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    max_train_steps = (args.epoch * len(train_loader)) // args.accumulation_steps

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=max_train_steps * 0.1,
        num_training_steps=max_train_steps
    )

    train_loader, eval_loader, model, optimizer, scheduler = accelerator.prepare(train_loader,
                                                                                 eval_loader,
                                                                                 model,
                                                                                 optimizer,
                                                                                 scheduler)

    progress_bar = tqdm(range(max_train_steps))

    overall_step = 0

    acc_best, f1_best = 0, 0
    total_time = 0.0
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.accumulation_steps

            accelerator.backward(loss)

            if step % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            overall_step += 1

            if (overall_step + 1) % args.print_step == 0:
                _lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(f"learning rate: {_lr:.3e}")
                print_loss = loss * args.accumulation_steps
                logger.info(f'loss: {print_loss.item():.5f}')
        epoch_end_time = time.time()
        total_time += epoch_end_time - epoch_start_time

        if epoch > args.start_eval_epoch:
            with torch.no_grad():
                results = eval_fn()
            logger.info(f" acc: {results['accuracy']:.3f}, macro f1: {results['f1']:.3f}")

            if f1_best < results['f1']:
                f1_best = results['f1']
                acc_best = results['accuracy']

            earlystop.update(results['f1'])
            if earlystop.stop:
                logger.info('-' * 30)
                logger.info(f"End at Epoch {epoch}")
                logger.info(f" Average training time of an epoch: {total_time / (epoch + 1)}")
                break

        logger.info('-' * 30)
        logger.info(f"End of Epoch {epoch}")

    with open(os.path.join(args.checkpoint, 'result.txt'), 'a', encoding='utf-8') as wf:
        wf.write(f"{acc_best:.3f}\t{f1_best:.3f}\n")

    logger.info(f" Best acc: {acc_best:.3f}, macro f1: {f1_best:.3f}")
