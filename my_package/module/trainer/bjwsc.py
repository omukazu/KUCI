import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from module.trainer.base import BaseTrainer


class BJWSCTrainer(BaseTrainer):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        reduction = 'sum' if mode == 'train' else 'none'
        ce_loss = F.binary_cross_entropy(output, batch['label'], reduction=reduction)
        return ce_loss

    def predict(self, output: torch.Tensor, _: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.where(output >= 0.5, 1., 0.)

    def compute_metrics(
        self,
        data_loader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple[float, dict[str, float]]:
        acc = torch.sum(predictions == labels).item() / len(example_ids)
        auc = roc_auc_score(*map(lambda x: x.to('cpu').detach().numpy().copy(), [labels, outputs]))
        return auc, {
            f'{data_loader.dataset.split}_acc': round(acc, 6),
            f'{data_loader.dataset.split}_auc': round(auc, 6)
        }
