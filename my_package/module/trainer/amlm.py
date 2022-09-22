import torch
import torch.nn.functional as F

from module.trainer.base import BaseTrainer


class AMLMTrainer(BaseTrainer):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor], _: str) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            output.logits.view(-1, self.model.module.config.vocab_size),
            batch['labels'].view(-1)
        )  # reduction='mean'
        return ce_loss

    def predict(self, *_):
        return ...

    def compute_metrics(self, *_):
        return ...
