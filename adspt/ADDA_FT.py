from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from utils import EarlyStop

from transformers import get_scheduler

from sklearn.metrics import accuracy_score, f1_score

from itertools import zip_longest
import os


@torch.no_grad()
def evaluating(encoder, classifier, eval_loader):
    # acc_metric = evaluate.load('accuracy')
    # f1_metric = evaluate.load('f1')

    labels, predictions = [], []
    encoder.eval()
    classifier.eval()
    for _, batch in enumerate(eval_loader):
        inputs = {k: batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
        inputs['use_prompt'] = True
        outputs = encoder(**inputs)
        logits = classifier(outputs['hidden'])
        preds = logits.argmax(dim=-1)

        predictions.extend(preds.detach().cpu().tolist())
        labels.extend(batch['labels'].detach().cpu().tolist())

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    results = {'accuracy': acc * 100, 'f1': f1 * 100}
    return results


def finetuning(encoder, classifier, train_loader, eval_loader, accelerator, args, logger, domain, model_return=True,
               recording=False):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)
    steps = args.ft_epoch * len(train_loader)

    # setup criterion and optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   list(encoder.parameters()) + list(classifier.parameters())), lr=args.lr)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=steps * 0.1,
        num_training_steps=steps
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    criterion = nn.CrossEntropyLoss()
    progress_bar = tqdm(range(steps))
    acc_best, f1_best = 0, 0
    for epoch in range(args.ft_epoch):
        encoder.train()
        classifier.train()
        for step, batch in enumerate(train_loader):
            inputs = {k: batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
            inputs['use_prompt'] = True
            outputs = encoder(**inputs)
            logits = classifier(outputs['hidden'])
            loss = criterion(logits.view(-1, args.class_num), batch['labels'].view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % args.print_step == 0:
                _lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(f"learning rate: {_lr:.3e}")
                logger.info(f'loss: {(loss.item()):.5f}')

            progress_bar.update(1)

        results = evaluating(encoder, classifier, eval_loader)
        logger.info(f" acc: {results['accuracy']:.3f}, macro f1: {results['f1']:.3f}")
        logger.info('=' * 30)
        logger.info(f"End of Epoch {epoch}")
        earlystop.update(results['accuracy'])

        if f1_best < results['f1']:
            f1_best = results['f1']
            acc_best = results['accuracy']

        if earlystop.stop:
            logger.info(f"Best results on {domain} test:")
            logger.info(f"Acc: {acc_best}, F1:{f1_best}")
            break

    if recording:
        with open(os.path.join(args.checkpoint, 'result.txt'), 'a', encoding='utf-8') as wf:
            wf.write(f"{acc_best:.3f}\t{f1_best:.3f}\n")

    if model_return:
        return encoder, classifier


def joint_finetuning(src_encoder, tgt_encoder, classifier, src_loader, tgt_loader, tgt_eval_loader, accelerator, args,
                     logger):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # instantiate EarlyStop
    earlystop = EarlyStop(args.patience)
    steps = max(len(tgt_loader), len(src_loader)) * args.ft_epoch
    # setup criterion and optimizer
    parameters = list(src_encoder.parameters()) + list(classifier.parameters()) + list(tgt_encoder.parameters())
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, parameters), lr=args.lr)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=steps * 0.1,
        num_training_steps=steps
    )

    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    criterion = nn.CrossEntropyLoss()
    progress_bar = tqdm(range(steps))
    acc_best, f1_best = 0, 0
    for epoch in range(args.ft_epoch):
        src_encoder.train()
        tgt_encoder.train()
        classifier.train()
        for step, (src_batch, tgt_batch) in enumerate(zip_longest(src_loader, tgt_loader)):
            optimizer.zero_grad()
            if tgt_batch is None:
                logger.info('empty target step')
                src_inputs = {k: src_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
                src_inputs['use_prompt'] = True
                outputs_src = src_encoder(**src_inputs)
                logits = classifier(outputs_src['hidden'])
                label = src_batch['labels']

            elif src_batch is None:
                logger.info('empty source step')
                tgt_inputs = {k: tgt_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
                tgt_inputs['use_prompt'] = True
                outputs_tgt = tgt_encoder(**tgt_inputs)
                logits = classifier(outputs_tgt['hidden'])
                label = tgt_batch['labels']

            else:
                src_inputs = {k: src_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
                src_inputs['use_prompt'] = True

                tgt_inputs = {k: tgt_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
                tgt_inputs['use_prompt'] = True

                outputs_src = src_encoder(**src_inputs)
                src_logits = classifier(outputs_src['hidden'])

                outputs_tgt = tgt_encoder(**tgt_inputs)
                tgt_logits = classifier(outputs_tgt['hidden'])

                logits = torch.cat((src_logits, tgt_logits), 0)

                label = torch.cat((src_batch['labels'], tgt_batch['labels']), 0)

            loss = criterion(logits.view(-1, args.class_num), label.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % args.print_step == 0:
                _lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(f"learning rate: {_lr:.3e}")
                logger.info(f'loss: {(loss.item()):.5f}')

            progress_bar.update(1)

        results = evaluating(tgt_encoder, classifier, tgt_eval_loader)
        logger.info(f" acc: {results['accuracy']:.3f}, macro f1: {results['f1']:.3f}")
        logger.info('=' * 30)
        logger.info(f"End of Epoch {epoch}")
        earlystop.update(results['f1'])

        if f1_best < results['f1']:
            f1_best = results['f1']
            acc_best = results['accuracy']

        if earlystop.stop:
            logger.info(f"Best results on target test:")
            logger.info(f"Acc: {acc_best}, F1:{f1_best}")
            break
