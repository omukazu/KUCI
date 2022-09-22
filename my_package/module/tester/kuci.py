import torch
import torch.nn.functional as F

from module.dataset.kuci import SentencePiece
from module.tester.base import BaseTester


class KUCITester(BaseTester):
    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ce_loss = F.cross_entropy(output, batch['label'], reduction='none')
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
            context_tokens = ' '.join(self.tokenizer.tokenize(example.context))
            choice_tokens = [
                ' '.join(self.tokenizer.tokenize(choice)) for choice in example.choices
            ]
            prediction = prediction.item()
            buf.append({
                'context': example.context,
                'context_tokens': context_tokens,
                'choices': example.choices,
                'choice_tokens': choice_tokens,
                'label': example.label,
                'output': list(map(lambda x: round(x, 3), output.tolist())),
                'prediction': prediction,
                'correct': example.label == prediction
            })
        return buf

    def preprocess(self, input_: str) -> tuple[list[list[str]]]:
        context, *choices = input_.split('@')
        if isinstance(self.tokenizer, SentencePiece):
            return [[context, choice] for choice in choices],
        else:
            context = ' '.join([morpheme.midasi for morpheme in self.jumanpp.analysis(context).mrph_list()])
            choices = map(
                lambda x: ' '.join(
                    [morpheme.midasi for morpheme in self.jumanpp.analysis(x).mrph_list()]
                ),
                choices
            )
            return [[context, choice] for choice in choices],
