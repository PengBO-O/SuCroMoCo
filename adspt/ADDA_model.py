import torch
from torch import nn


class SoftPromptLM(nn.Module):
    def __init__(self, base_model, pooling='cls', prompt_len=10):
        super(SoftPromptLM, self).__init__()
        self.backbone = base_model
        self.config = base_model.config
        self.pooling = pooling
        self.prompt_len = prompt_len
        self.prompt_embed = torch.nn.Embedding(prompt_len + 1, self.config.hidden_size)
        torch.nn.init.xavier_normal_(self.prompt_embed.weight)

        self.dropout = torch.nn.Dropout(0.2)

        self.pseudo_tokens = torch.arange(1, prompt_len + 1).long()

    def soft_prompt_embeddings(self, batch_size):
        batch_prompt = self.pseudo_tokens.unsqueeze(0).expand(batch_size, -1).cuda()
        prompt_embeds = self.prompt_embed(batch_prompt)
        return prompt_embeds

    def forward(self, **kwargs):

        if bool(kwargs['use_prompt']):
            batch_size = kwargs['input_ids'].shape[0]

            raw_embedding = self.backbone.embeddings(
                input_ids=kwargs['input_ids'],
            )
            prompt_embeds = self.soft_prompt_embeddings(batch_size=batch_size)
            temp = []
            for i in range(batch_size):
                sep_loc = kwargs['sep_locations'][i][1].item()
                no_sep_embed = torch.cat((raw_embedding[i][:sep_loc, :], prompt_embeds[i]))
                input_embed = torch.cat((no_sep_embed, raw_embedding[i][sep_loc:, :]))
                temp.append(input_embed)
            inputs_embeds = torch.stack(temp, dim=0)
            # cls_embeds, rest_embeds = torch.split(raw_embedding, [1, kwargs['input_ids'].shape[1] - 1], dim=1)
            # prompt_embeds.requires_grad = True
            # no_cls_embeds = torch.cat((prompt_embeds, rest_embeds), dim=1)
            # inputs_embeds = torch.cat((cls_embeds, no_cls_embeds), dim=1)
            prompt_attention_mask = torch.ones(batch_size, self.prompt_len).to(self.backbone.device)
            attention_mask = torch.cat((prompt_attention_mask, kwargs['attention_mask']), dim=1)

            raw_outputs = self.backbone(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )

        else:
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

        outputs = {'hidden': pooled_output}

        if 'locations' in kwargs.keys():
            mask_hidden = torch.stack(
                [last_hidden[i, kwargs['locations'][i], :] for i in range(kwargs['locations'].shape[0])]
            )
            outputs['mask_hidden'] = mask_hidden

        return outputs


class Classifier(nn.Module):
    def __init__(self, dropout=0.2, num_class=2):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, num_class))

    def forward(self, x):
        out = self.net(x)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(768, 300),
            nn.ReLU(),
            nn.Linear(300, 2)
        )


    def forward(self, x):
        """Forward the discriminator."""
        out = self.net(x)
        return out
