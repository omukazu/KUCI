import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertForSJWSC(BertPreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.bert = BertModel(cfg, add_pooling_layer=False)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.post_init()

    def forward(
        self,
        inputs: dict[str, torch.LongTensor],
        pronoun_map: torch.Tensor,
        antecedent1_map: torch.Tensor,
        antecedent2_map: torch.Tensor,
        **_
    ) -> torch.Tensor:
        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(**inputs)
        batch_seq = encoded.last_hidden_state  # batch_size, max_seq_len, hidden_size
        pronoun_vectors, antecedent1_vectors, antecedent2_vectors = map(
            lambda x: torch.sum(batch_seq * x.unsqueeze(2), dim=1) / torch.sum(x, dim=1, keepdim=True),
            [pronoun_map, antecedent1_map, antecedent2_map]
        )
        output = torch.stack(
            [
                self.metric(pronoun_vectors, antecedent1_vectors),
                self.metric(pronoun_vectors, antecedent2_vectors)
            ],
            dim=1
        )  # batch_size, 2
        return output


class XLMRobertaForSJWSC(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    config_class = XLMRobertaConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.config = cfg

        self.roberta = XLMRobertaModel(cfg, add_pooling_layer=False)
        self.metric = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.post_init()

    def forward(
        self,
        inputs: dict[str, torch.LongTensor],
        pronoun_map: torch.Tensor,
        antecedent1_map: torch.Tensor,
        antecedent2_map: torch.Tensor,
        **_
    ) -> torch.Tensor:
        encoded: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(**inputs)
        batch_seq = encoded.last_hidden_state  # batch_size, max_seq_len, hidden_size
        pronoun_vectors, antecedent1_vectors, antecedent2_vectors = map(
            lambda x: torch.sum(batch_seq * x.unsqueeze(2), dim=1) / torch.sum(x, dim=1, keepdim=True),
            [pronoun_map, antecedent1_map, antecedent2_map]
        )
        output = torch.stack(
            [
                self.metric(pronoun_vectors, antecedent1_vectors),
                self.metric(pronoun_vectors, antecedent2_vectors)
            ],
            dim=1
        )  # batch_size, 2
        return output
