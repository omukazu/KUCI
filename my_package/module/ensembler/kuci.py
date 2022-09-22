import torch

from module.ensembler.base import BaseBlender


class KUCIBlender(BaseBlender):
    def predict(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=1)

    def compute_metrics(
        self,
        data_loader,
        example_ids: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, ...]:
        num_correct = torch.sum(self.predict(outputs) == labels)
        acc = (num_correct / len(example_ids)).item()
        return {
            f'{data_loader.dataset.split}_acc': round(acc, 6),
            f'{data_loader.dataset.split}_details': {
                'num_correct': num_correct.item(),
                'num_examples': len(example_ids),
            }
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
