from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.mc_head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(cfg.hidden_dropout_prob)),
                ('fc', nn.Linear(cfg.hidden_size, 1))
            ])
        )

        self.post_init()

    def forward(self, inputs: dict[str, torch.LongTensor], **_) -> torch.FloatTensor:
        batch_size, num_choices, max_seq_len = inputs['input_ids'].shape
        inputs = {key: value.view(batch_size * num_choices, max_seq_len) for key, value in inputs.items()}

        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**inputs)
        cls = encoded.last_hidden_state[:, 0, :]
        output = self.mc_head(cls).squeeze(1).view(batch_size, num_choices)
        return output


class XLMRobertaForMultipleChoice(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = XLMRobertaConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.roberta = XLMRobertaModel(cfg, add_pooling_layer=False)
        self.mc_head = nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(cfg.hidden_dropout_prob)),
                ('fc', nn.Linear(cfg.hidden_size, 1))
            ])
        )

        self.post_init()

    def forward(self, inputs: dict[str, torch.LongTensor], **_) -> torch.FloatTensor:
        batch_size, num_choices, max_seq_len = inputs['input_ids'].shape
        inputs = {key: value.view(batch_size * num_choices, max_seq_len) for key, value in inputs.items()}

        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(**inputs)
        cls = encoded.last_hidden_state[:, 0, :]
        output = self.mc_head(cls).squeeze(1).view(batch_size, num_choices)
        return output
