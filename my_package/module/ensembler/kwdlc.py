import torch

from module.ensembler.base import BaseBlender


class KWDLCBlender(BaseBlender):
    def predict(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=1)

    def compute_metrics(
        self,
        data_loader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, ...]:
        predictions = self.predict(outputs)

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

        return {
            f'{data_loader.dataset.split}_acc': round(acc, 6),
            f'{data_loader.dataset.split}_prec': round(prec, 6),
            f'{data_loader.dataset.split}_rec': round(rec, 6),
            f'{data_loader.dataset.split}_f1': round(f1, 6),
            f'{data_loader.dataset.split}_details': details
        }

    def error_analysis(
        self,
        data_loader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor
    ) -> list:
        examples = [data_loader.dataset.examples[example_id] for example_id in example_ids]
        predictions = self.predict(outputs)

        buf = []
        for example, output, prediction in zip(examples, outputs, predictions):
            clause_tokens1 = ' '.join(self.tokenizer.tokenize(example.clause1))
            clause_tokens2 = ' '.join(self.tokenizer.tokenize(example.clause2))
            prediction = prediction.item()
            buf.append({
                'clause1': example.clause1,
                'clause_tokens1': clause_tokens1,
                'clause2': example.clause2,
                'clause_tokens2': clause_tokens2,
                'label': example.label,
                'output': list(map(lambda x: round(x, 3), output.tolist())),
                'prediction': prediction,
                'correct': example.label == prediction
            })
        return buf
