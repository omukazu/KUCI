import torch
import torch.nn.functional as F

from module.trainer.base import BaseTrainer


class KWDLCTrainer(BaseTrainer):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        reduction = 'sum' if mode == 'train' else 'none'
        ce_loss = F.cross_entropy(output, batch['label'], reduction=reduction)
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
        details = {'tp': [], 'p': [], 't': []}
        for idx, cls in enumerate(data_loader.dataset.idx2cls):
            if cls == '談話関係なし':
                continue
            details['tp'].append(torch.sum((predictions == idx) & (labels == idx)).item())
            details['p'].append(torch.sum(predictions == idx).item())
            details['t'].append(torch.sum(labels == idx).item())
        tp, p, t = map(lambda x: sum(details[x]), ['tp', 'p', 't'])

        acc = (torch.sum(predictions == labels) / len(example_ids)).item()
        prec = tp / p if p > 0 else 0.
        rec = tp / t if t > 0 else 0.
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.

        return f1, {
            f'{data_loader.dataset.split}_acc': round(acc, 6),
            f'{data_loader.dataset.split}_prec': round(prec, 6),
            f'{data_loader.dataset.split}_rec': round(rec, 6),
            f'{data_loader.dataset.split}_f1': round(f1, 6),
            f'{data_loader.dataset.split}_details': details
        }
