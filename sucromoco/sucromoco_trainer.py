from tqdm.auto import tqdm
import time
import torch
from torch.optim import AdamW

from accelerate import Accelerator
from transformers import get_scheduler

from itertools import zip_longest
from sklearn.metrics import accuracy_score, f1_score

import os
from utils import EarlyStop


class CrossAdaptTrainer:
    def __init__(self, model, model_aug, contrast,
                 fin_loader, pro_loader,
                 fin_eval_loader,
                 logger, args, accelerator=None):
        super(CrossAdaptTrainer).__init__()
        self.logger = logger
        self.args = args
        if accelerator:
            self.accelerator = accelerator
        else:
            self.accelerator = Accelerator()
        self.acc_best, self.f1_best = 0, 0
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.model, self.model_aug, self.contrast, \
            self.fin_loader, self.pro_loader, \
            self.fin_eval_loader = self.accelerator.prepare(model, model_aug, contrast,
                                                            fin_loader, pro_loader,
                                                            fin_eval_loader)

    @torch.no_grad()
    def momentum_encoder_update(self):
        for param_q, param_k in zip(self.model.parameters(), self.model_aug.parameters()):
            param_k.data = param_k.data * self.args.m + param_q.data * (1. - self.args.m)


    @torch.no_grad()
    def evaluating(self, eval_loader):

        labels, predictions = [], []
        self.model.eval()
        for _, batch in enumerate(eval_loader):
            if self.args.prompt:
                inputs = {k: batch[k] for k in ['input_ids', 'attention_mask', 'locations']}
            else:
                inputs = {k: batch[k] for k in ['input_ids', 'attention_mask']}
            model_outputs = self.model(**inputs)
            preds = model_outputs['logits'].argmax(dim=-1)

            predictions.extend(preds.detach().cpu().tolist())
            labels.extend(batch['labels'].detach().cpu().tolist())

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        results = {'accuracy': acc * 100, 'f1': f1 * 100}
        return results

    def recording(self, results, name='prototype'):
        self.logger.info(f'evaluation results of {name} set')
        self.logger.info(f" acc: {results['accuracy']:.3f}, macro f1: {results['f1']:.3f}")
        self.logger.info('-' * 30)
        return

    def best_result(self):
        with open(os.path.join(self.args.checkpoint, 'result.txt'), 'a', encoding='utf-8') as wf:
            wf.write(f"{self.acc_best:.3f}\t{self.f1_best:.3f}\n")
        self.logger.info('Classification training finished')
        self.logger.info(f" Best acc: {self.acc_best:.3f}, macro f1: {self.f1_best:.3f}")
        self.logger.info(f" Done!!!!!")
        return

    def step_progress(self, batch, contrast_against: str = 'prototype'):
        outputs = self.model(**batch)
        logits = outputs['logits']
        cls_loss = self.ce_criterion(logits.view(-1, self.args.class_num), batch['labels'].view(-1))
        cl_loss = self.contrast(outputs['hidden'], batch['labels'], adapt_to=contrast_against)

        loss = cls_loss + cl_loss
        outputs['loss'] = loss
        with torch.no_grad():
            aug_outputs = self.model_aug(**batch)

        return outputs, aug_outputs

    def training(self):
        earlystop = EarlyStop(patience=self.args.patience)
        steps = self.args.epoch * max(len(self.fin_loader), len(self.pro_loader))
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=steps * 0.1,
            num_training_steps=steps
        )

        progress_bar = tqdm(range(steps))
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        total_time = 0.0

        for epoch in range(self.args.epoch):
            epoch_start_time = time.time()

            self.model.train()
            self.model_aug.eval()

            for step, (fin_batch, pro_batch) in enumerate(zip_longest(self.fin_loader, self.pro_loader)):
                if fin_batch is None:
                    self.logger.info('empty financial step')
                    pro_outputs, aug_pro_outputs = self.step_progress(pro_batch, contrast_against='finance')
                    loss = pro_outputs['loss']

                    self.contrast.enqueue_dequeue(aug_pro_outputs['hidden'], pro_batch['labels'],
                                                  memory_type='prototype')
                elif pro_batch is None:
                    self.logger.info('empty prototype step')
                    self.logger.info(f"{fin_batch['input_ids'].shape[0]} samples from target")
                    fin_outputs, aug_fin_outputs = self.step_progress(fin_batch, contrast_against='prototype')
                    loss = fin_outputs['loss']
                    self.contrast.enqueue_dequeue(aug_fin_outputs['hidden'], fin_batch['labels'],
                                                  memory_type='finance')

                else:
                    pro_outputs, aug_pro_outputs = self.step_progress(pro_batch, contrast_against='finance')
                    fin_outputs, aug_fin_outputs = self.step_progress(fin_batch, contrast_against='prototype')

                    loss = pro_outputs['loss'] + fin_outputs['loss']

                    self.contrast.enqueue_dequeue(aug_pro_outputs['hidden'], pro_batch['labels'],
                                                  memory_type='prototype')
                    self.contrast.enqueue_dequeue(aug_fin_outputs['hidden'], fin_batch['labels'],
                                                  memory_type='finance')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if (step + 1) % self.args.print_step == 0:
                    _lr = optimizer.state_dict()['param_groups'][0]['lr']
                    self.logger.info(f"learning rate: {_lr:.3e}")
                    self.logger.info(f'loss: {(loss.item()):.5f}')

                self.momentum_encoder_update()

                progress_bar.update(1)

            epoch_end_time = time.time()
            total_time += epoch_end_time - epoch_start_time
            fin_eval_results = self.evaluating(eval_loader=self.fin_eval_loader)
            self.recording(fin_eval_results, name='target')

            if self.f1_best < fin_eval_results['f1']:
                self.f1_best = fin_eval_results['f1']
                self.acc_best = fin_eval_results['accuracy']

            self.logger.info('=' * 30)
            self.logger.info(f"End of Epoch {epoch}")

            earlystop.update(fin_eval_results['f1'])
            if earlystop.stop:
                self.logger.info(f" Average training time of an epoch: {total_time / (epoch + 1)}")

                break

        self.best_result()
