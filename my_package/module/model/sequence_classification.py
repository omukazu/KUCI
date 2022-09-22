from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.cls_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(cfg.hidden_dropout_prob)),
            ('fc', nn.Linear(cfg.hidden_size, cfg.num_classes))
        ]))

        self.post_init()

    def forward(self, inputs: dict[str, torch.LongTensor], **_) -> torch.FloatTensor:
        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**inputs)
        cls = encoded.last_hidden_state[:, 0, :]
        output = self.cls_head(cls)
        return output


class XLMRobertaForSequenceClassification(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = XLMRobertaConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.roberta = XLMRobertaModel(cfg, add_pooling_layer=False)
        self.cls_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(cfg.hidden_dropout_prob)),
            ('fc', nn.Linear(cfg.hidden_size, cfg.num_classes))
        ]))

        self.post_init()

    def forward(self, inputs: dict[str, torch.LongTensor], **_) -> torch.FloatTensor:
        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(**inputs)
        cls = encoded.last_hidden_state[:, 0, :]
        output = self.cls_head(cls)
        return output
