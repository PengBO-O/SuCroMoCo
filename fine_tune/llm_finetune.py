import pathlib
import sys
import time
import argparse
from loguru import logger

import datasets
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType, PeftType

sys.path.append("..")
from utils import load_data, loader_process, train_fn


def parser_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-2-7b-hf',
                        choices=['meta-llama/Llama-2-7b-hf', 'ChanceFocus/finma-7b-nlp'])

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "int8"],
        help="Whether to use mixed precision. We can only afford int8 and fp16"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )

    parser.add_argument('--dataset', type=str, default='tweetfinsent',
                        choices=["fpb", "stocksen", "semeval", "tweetfinsent"])
    parser.add_argument("--joint", default=False, type=lambda x: (str(x).lower() in ['true', 1, 'yes']))
    parser.add_argument("--sample", default=False, type=lambda x: (str(x).lower() in ['true', 1, 'yes']))
    parser.add_argument('--shots', type=int, default=2000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument('--start_eval_epoch', type=int, default=5)

    _args_ = parser.parse_args()

    return _args_


def preparation(args):
    repo_root = pathlib.Path(__file__).parent.parent
    args.data_path = repo_root / 'data'
    args.time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if args.joint:
        args.checkpoint = repo_root.joinpath(
            f"/save/LLM_PEFT/LoRA/joint/{args.dataset}/{args.model_id.split('/')[-1]}/")
    else:
        args.checkpoint = repo_root.joinpath(f"/save/LLM_PEFT/LoRA/sole/{args.dataset}/{args.model_id.split('/')[-1]}/")

    logger.add(args.checkpoint.joinpath(f'{args.time}_log.txt'))

    logger.info('llm_finetune.py')
    logger.info(f'Backbone model: {args.model_id}')
    logger.info(f'Epochs: {args.epoch}')
    logger.info(f'total BS: {args.batch_size}')
    logger.info(f'Learning rate: {args.lr:.3e}')

    logger.info(f'Dataset: {args.dataset}')
    args.num_labels = 3

    logger.info(f"Precision: int8")
    logger.info(f"Number of labels: {args.num_labels}")

    if args.model_id == 'meta-llama/Llama-2-7b-hf':
        args.hf_token = ""
    else:
        args.hf_token = None


def main(args):
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

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    peft_type = PeftType.LORA
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             inference_mode=False,
                             r=8,
                             lora_alpha=16,
                             lora_dropout=0.1,
                             target_modules=["q_proj", "v_proj"])

    logger.info(f"Financial training samples: {len(fin_data)}")
    train_data = {}
    if args.joint:
        for k in fin_data.keys():
            train_data[k] = fin_data[k]+pro_data[k]
    else:
        train_data = fin_data

    logger.info(f"Total Training samples: {len(train_data)}")

    train_dataset = datasets.Dataset.from_dict(train_data, split=datasets.Split.TRAIN)
    eval_dataset = datasets.Dataset.from_dict(fin_eval_data, split=datasets.Split.TEST)

    max_char = max(train_dataset.to_pandas()['text'].str.len().max(),
                   eval_dataset.to_pandas()['text'].str.len().max())
    logger.info(f'max_char: {max_char}')
    max_words = max(
        train_dataset.to_pandas()['text'].str.split().apply(lambda x: max(len(sentence) for sentence in x)).max(),
        eval_dataset.to_pandas()['text'].str.split().apply(lambda x: max(len(sentence) for sentence in x)).max())
    logger.info(f'max_words: {max_words}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_id,
                                              padding_side="left",
                                              token=args.hf_token)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id,
                                                               device_map='auto',
                                                               # load_in_8bit=True,
                                                               token=args.hf_token,
                                                               num_labels=args.num_labels,
                                                               return_dict=True)

    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()

    logger.info(f'f"trainable params: {trainable_params:,d} || all params: {all_param:,d}')
    logger.info(f'trainable%: {100 * trainable_params / all_param}')

    train_dataloader = loader_process(args, tokenizer, train_dataset, accelerator)
    args.print_step = len(train_dataloader) // 3

    eval_dataloader = loader_process(args, tokenizer, eval_dataset, accelerator)
    args.print_step = len(train_dataloader) // 3

    train_fn(args=args,
             train_loader=train_dataloader, eval_loader=eval_dataloader,
             model=model,
             accelerator=accelerator,
             logger=logger)


if __name__ == '__main__':
    args = parser_args()
    preparation(args)
    main(args)
