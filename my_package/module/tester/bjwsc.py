import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from module.dataset.bjwsc import SentencePiece
from module.tester.base import BaseTester


class BJWSCTester(BaseTester):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(output, batch['label'], reduction='none')
        return bce_loss

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
        acc = (torch.sum(predictions == labels) / len(example_ids)).item()
        auc = roc_auc_score(*map(lambda x: x.to('cpu').detach().numpy().copy(), [labels, outputs]))
        return acc, {
            f'{data_loader.dataset.split}_acc': round(acc, 6),
            f'{data_loader.dataset.split}_auc': round(auc, 6)
        }

    def error_analysis(
        self,
        data_loader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        predictions: torch.Tensor
    ) -> list:
        examples = [data_loader.dataset.examples[example_id] for example_id in example_ids]

        buf = []
        for example, output, prediction in zip(examples, outputs, predictions):
            input_tokens = ' '.join(self.tokenizer.tokenize(example.input))
            prediction = prediction.item()
            buf.append({
                'input': example.input,
                'input_tokens': input_tokens,
                'label': example.label,
                'output': round(output.item(), 3),
                'prediction': prediction,
                'correct': example.label == prediction
            })
        return buf

    def preprocess(self, input_: str) -> tuple[str]:
        if isinstance(self.tokenizer, SentencePiece):
            return input_,
        else:
            return ' '.join(morpheme.midasi for morpheme in self.jumanpp.analysis(input_).mrph_list()),
