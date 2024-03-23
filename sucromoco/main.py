import pathlib
import time
import argparse
import copy
import sys

import torch

from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from sucromoco_trainer import CrossAdaptTrainer
from moco import CrossMoCo
from accelerate import Accelerator

sys.path.append("..")
from models import PooledLM
from utils import load_data, MyDataset, encoding


def parser_args():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--base_model', type=str, default='bert-base-uncased',
                        choices=['bert-base-uncased', 'roberta-base',
                                 'SALT-NLP/FLANG-BERT', 'SALT-NLP/FLANG-Roberta'])
    parser.add_argument('--pooler', type=str, default='mean', choices=['cls', 'mean', 'pooled'])
    parser.add_argument('--dataset', type=str, default='tweetfinsent',
                        choices=["fpb", "stocksen", "semeval", "tweetfinsent"])

    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--m", type=float, default=0.999, help='update weight of augmentation model')
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--k", type=int, default=1)

    _args_ = parser.parse_args()

    return _args_


def preparation():
    global args
    repo_root = pathlib.Path(__file__).parent.parent
    args.data_path = repo_root / 'data'
    args.time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    args.checkpoint = repo_root.joinpath(f"/save/sucromoco/{args.base_model}/{args.dataset}/")
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    logger.add(args.checkpoint.joinpath(f'{args.time}_log.txt'))

    logger.info(f'backbone: {args.base_model}')
    logger.info(f"pooling type: {args.pooler}")
    logger.info(f'epochs: {args.epoch}')
    logger.info(f'total batch size: {args.batch_size}')
    logger.info(f'learning rate: {args.lr:.3e}')
    logger.info(f'temperature: {args.temperature}')
    logger.info(f'm: {args.m}')
    logger.info(f'dataset: {args.dataset}')
    if args.dataset == 'stocksen':
        args.class_num = 2
    else:
        args.class_num = 3
    args.print_step = 50


@torch.no_grad()
def get_memories(model, loader):
    global args
    accelerate = Accelerator()
    model, loader = accelerate.prepare(model, loader)
    hiddens, labels = [], []
    model.eval()
    for _, batch in enumerate(loader):
        inputs = {k: batch[k] for k in ('input_ids', 'attention_mask')}
        outputs = model(**inputs)
        hiddens.append(outputs['hidden'])
        labels.append(batch['labels'])
    out_items = {'hidden': torch.cat(hiddens, dim=0).detach().cpu(), 'labels': torch.cat(labels, dim=0).detach().cpu()}

    return out_items


def loader_building(data, tokenizer, bs: int, do_shuffle=False):
    global args

    encodings = encoding(data, tokenizer=tokenizer)
    loader = DataLoader(MyDataset(encodings),
                        batch_size=bs, shuffle=do_shuffle)
    return loader


def main():
    global args
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    if args.class_num == 2:
        pro_data = load_data(f"{args.data_path}/sst2_train.csv")
    else:
        pro_data = load_data(f"{args.data_path}/fpb_allagree.csv")

    if args.dataset in ['stocksen', 'tweetfinsent']:
        fin_data = load_data(f"{args.data_path}/{args.dataset}_train.csv")
        fin_eval_data = load_data(f"{args.data_path}/{args.dataset}_test.csv")
    elif args.dataset in ['semevel', 'fpb']:
        fin_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_train_{args.k}.csv")
        fin_eval_data = load_data(f"{args.data_path}/{args.dataset}_5fold/{args.dataset}_test_{args.k}.csv")

    else:
        raise ValueError("No such dataset")

    logger.info(f"{len(pro_data['text'])} samples from prototype")
    logger.info(f"{len(fin_data['text'])} samples from finance")

    pro_per = len(pro_data) / (len(pro_data) + len(fin_data))
    fin_per = 1 - pro_per

    pro_loader = loader_building(pro_data, tokenizer=tokenizer,
                                 bs=int(round(pro_per * args.batch_size)),
                                 do_shuffle=True)
    fin_eval_loader = loader_building(fin_eval_data, tokenizer=tokenizer,
                                      bs=args.batch_size,
                                      do_shuffle=False)
    fin_loader = loader_building(fin_data, tokenizer=tokenizer,
                                 bs=int(round(fin_per * args.batch_size)),
                                 do_shuffle=True)
    logger.info(f"prototype dataloader length {len(pro_loader)}")
    logger.info(f"finance dataloader length {len(fin_loader)}")

    base_model = AutoModel.from_pretrained(args.base_model)
    model = PooledLM(base_model,
                     pooling=args.pooler,
                     num_labels=args.class_num,
                     )
    model_aug = copy.deepcopy(model)

    pro_memory_items = get_memories(model=model, loader=pro_loader)
    fin_memory_items = get_memories(model=model, loader=fin_loader)

    contrast = CrossMoCo(fin_memory=fin_memory_items, pro_memory=pro_memory_items, args=args)

    trainer = CrossAdaptTrainer(model=model, model_aug=model_aug, contrast=contrast,
                                fin_loader=fin_loader, pro_loader=pro_loader,
                                fin_eval_loader=fin_eval_loader,
                                logger=logger, args=args)

    trainer.training()

    return


if __name__ == '__main__':
    args = parser_args()
    preparation()
    main()
