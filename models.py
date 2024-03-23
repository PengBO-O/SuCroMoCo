import torch

class PooledLM(torch.nn.Module):
    def __init__(self, base_model, num_labels=2, pooling='cls'):
        super(PooledLM, self).__init__()
        self.backbone = base_model
        self.config = base_model.config
        self.num_labels = num_labels
        self.pooling = pooling

        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **kwargs):

        inputs = {k: kwargs[k] for k in ('input_ids', 'attention_mask')}
        raw_outputs = self.backbone(**inputs)
        last_hidden = raw_outputs.last_hidden_state

        if self.pooling == 'pooled':
            pooled_output = raw_outputs.pooler_output
        elif self.pooling == 'cls':
            pooled_output = last_hidden[:, 0, :]
        elif self.pooling == 'mean':
            pooled_output = last_hidden.mean(dim=1)
        else:
            raise ValueError('pooling mode must in pooled, cls or mean')

        logits = self.classifier(self.dropout(pooled_output))

        outputs = {'hidden': pooled_output,
                   'logits': logits}

        return outputs


class DualLM(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(DualLM, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, *args, **kwargs):
        raw_outputs = self.base_model(*args, **kwargs)
        hiddens = raw_outputs.last_hidden_state
        cls_feats = hiddens[:, 0, :]
        label_feats = hiddens[:, 1:self.num_classes + 1, :]
        predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs
