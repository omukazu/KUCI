from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast

from utils import tqdm, ObjectHook


SentencePiece = (XLMRobertaTokenizer, XLMRobertaTokenizerFast)


@dataclass
class AMLMExample:
    example_id: int
    input: str


class AMLM(Dataset):
    def __init__(
        self,
        input_path: Path,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        debug: bool = False,
        confirm_inputs: bool = False,
        **_
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.debug = debug

        self.examples = self._load_examples(input_path, confirm_inputs=confirm_inputs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        inputs = self.tokenizer(
            *self._get_input_text(example),
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        length = torch.sum(inputs['attention_mask']).item()

        if isinstance(self.tokenizer, SentencePiece):
            inputs['token_type_ids'] = torch.LongTensor([0] * self.max_seq_len)

        return {
            'example_id': example.example_id,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'token_type_idex': inputs['token_type_ids'],
            'special_tokens_mask': inputs['special_tokens_mask'],
            'length': length,
        }

    def _load_examples(self, input_path: Path, confirm_inputs: bool = False):
        examples = []
        example_id = 0
        with input_path.open(mode='r') as f:
            for line in tqdm(f):
                old = ' ' if isinstance(self.tokenizer, SentencePiece) else ''
                example = AMLMExample(
                    example_id=example_id,
                    input=line.strip().replace(old, '')
                )
                examples.append(example)
                example_id += 1
                if self.debug and len(examples) == 10:
                    break

        if confirm_inputs:
            print('*** confirm inputs ***')
            for i in range(1, 4):
                self.confirm_inputs(examples[-i])

        return examples

    def confirm_inputs(self, example: AMLMExample) -> None:
        inputs = self.tokenizer(
            *self._get_input_text(example),
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        inputs = {key: value.squeeze(0).tolist() for key, value in inputs.items()}

        input_keys = inputs.keys()
        if 'token_type_ids' not in input_keys:
            inputs['token_type_ids'] = ...
        if keys := (input_keys - {'input_ids', 'attention_mask', 'token_type_ids'}):
            print(f'unexpected keys: {keys}')

        values = [
            ' '.join(self.tok.convert_ids_to_tokens(inputs['input_ids'])),
            inputs['attention_mask'],
            inputs['token_type_ids']
        ]
        pprint(values, width=max(len(str(value)) for value in values) + 7)

    @staticmethod
    def _get_input_text(example: AMLMExample) -> tuple[str]:
        return example.input,
