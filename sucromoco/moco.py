import torch
from torch import nn
import torch.nn.functional as F


def matrix_sim(q, memory):
    return F.cosine_similarity(q.unsqueeze(1), memory.clone().detach().unsqueeze(0), dim=-1)


class SupeConMemoryLoss(nn.Module):
    def __init__(self, temperature):
        super(SupeConMemoryLoss, self).__init__()
        self.temperature = temperature

    def forward(self, q, labels, memory_logits, memory_targets):
        src_sims = matrix_sim(q, q)
        src_contrast = torch.div(src_sims, self.temperature)
        src_logits_max, _ = torch.max(src_contrast, dim=1, keepdim=True)
        src_logits = src_contrast - src_logits_max.detach()

        src_targets = torch.eq(labels, labels.unsqueeze(1)).to(torch.float).cuda()
        src_logits_mask = torch.scatter(
            torch.ones_like(src_targets),
            1,
            torch.arange(q.shape[0]).view(-1, 1).cuda(),
            0
        )

        src_targets = src_targets * src_logits_mask
        targets = torch.cat((src_targets, memory_targets), dim=1)

        src_exp_logits = torch.exp(src_logits) * src_logits_mask
        logits = torch.cat((src_logits, memory_logits), dim=1)

        exp_logits = torch.cat((src_exp_logits, torch.exp(memory_logits)), dim=1)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (targets * log_prob).sum(1) / targets.sum(1)
        loss = -1 * mean_log_prob_pos
        loss = loss.view(1, q.shape[0]).mean()

        return loss


class CrossMoCo(nn.Module):
    def __init__(self, fin_memory, pro_memory, args):
        # Keep memory unchanged
        super(CrossMoCo, self).__init__()
        self.args = args

        self.register_buffer('pro_memory', pro_memory['hidden'])
        self.register_buffer('fin_memory', fin_memory['hidden'])

        self.register_buffer('pro_labels', pro_memory['labels'])
        self.register_buffer('fin_labels', fin_memory['labels'])

        self.pro_index = 0
        self.fin_index = 0

        self.pro_queue_size = self.pro_labels.shape[0]
        self.fin_queue_size = self.fin_labels.shape[0]

        self.supcon_criterion = SupeConMemoryLoss(temperature=self.args.temperature)
        self.ce_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    @torch.no_grad()
    def enqueue_dequeue(self, q, label, memory_type):
        bs = q.shape[0]
        out_ids = torch.arange(bs).cuda()
        if memory_type == 'prototype':
            out_ids += self.pro_index
            out_ids = torch.fmod(out_ids, self.pro_queue_size)
            out_ids = out_ids.long()
            self.pro_memory.index_copy_(0, out_ids, q)
            self.pro_labels.index_copy_(0, out_ids, label)
            self.pro_index = (self.pro_index + bs) % self.pro_queue_size
        elif memory_type == 'finance':
            out_ids += self.fin_index
            out_ids = torch.fmod(out_ids, self.fin_queue_size)
            out_ids = out_ids.long()
            self.fin_memory.index_copy_(0, out_ids, q)
            self.fin_labels.index_copy_(0, out_ids, label)
            self.fin_index = (self.fin_index + bs) % self.fin_queue_size

        else:
            raise ValueError('Memory type must be finance or prototype!')
        return

    def forward(self, q, labels, adapt_to: str = 'prototype'):

        """
        since we calculate the similarity between different domains,
        so that the q need to be taken into account
        """

        if adapt_to == 'prototype':
            memory_sims = matrix_sim(q, self.pro_memory)
            memory_targets = (labels[:, None] == self.pro_labels[None, :]).to(torch.float).cuda()
        elif adapt_to == 'finance':
            memory_sims = matrix_sim(q, self.fin_memory)
            memory_targets = (labels[:, None] == self.fin_labels[None, :]).to(torch.float).cuda()
        # sim_logits = torch.einsum('nc,ck->nk', [q, self.memory.clone().detach().transpose(1, 0)])
        else:
            raise ValueError('only have prototype and finance memory')

        memory_logits = torch.div(memory_sims, self.args.temperature)

        loss = self.supcon_criterion(q, labels, memory_logits, memory_targets)

        return loss
