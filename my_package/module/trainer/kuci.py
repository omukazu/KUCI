import torch
import torch.nn.functional as F

from module.trainer.base import BaseTrainer


class KUCITrainer(BaseTrainer):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        ce_loss = F.cross_entropy(output, batch['label'], reduction='none')
        if mode == 'train':
            ce_loss = torch.sum(ce_loss * batch['weight'])
        return ce_loss

    def predict(self, output: torch.Tensor, _: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.argmax(output, dim=1)

    def compute_metrics(
        self,
        data_loader,
        example_ids: torch.Tensor,
        _: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[float, dict[str, float]]:
        acc = (torch.sum(predictions == labels) / len(example_ids)).item()
        return acc, {f'{data_loader.dataset.split}_acc': round(acc, 6)}
