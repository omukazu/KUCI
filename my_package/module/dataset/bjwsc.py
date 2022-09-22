import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast

from utils import tqdm, ObjectHook


SentencePiece = (XLMRobertaTokenizer, XLMRobertaTokenizerFast)


@dataclass
class BJWSCExample:
    example_id: int
    input: str
    label: float


class BJWSC(Dataset):
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
            example.input,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        length = torch.sum(inputs['attention_mask']).item()

        return {
            'example_id': example.example_id,
            'inputs': inputs,
            'length': length,
            'label': float(example.label),
        }

    def _load_examples(self, input_path: Path, confirm_inputs: bool = False) -> list[BJWSCExample]:
        examples = []
        example_id = 0
        with input_path.open(mode='r') as f:
            for idx, line in enumerate(tqdm(f)):
                example = json.loads(line, object_hook=ObjectHook)
                example.input = self._preprocess(example)
                example = BJWSCExample(
                    example_id=example_id,
                    input=example.input,
                    label=example.label
                )
                examples.append(example)
                example_id += 1
                if self.debug and len(examples) == 10:
                    break

        if confirm_inputs:
            print('*** confirm inputs ***')
            for i in range(1, 2):
                self.confirm_inputs(examples[-i])

        return examples

    def _preprocess(self, example: BJWSCExample) -> str:
        old = ' ' if isinstance(self.tokenizer, SentencePiece) else ''
        sent1, sent2 = example.input
        if sent2.endswith(' だ 。'):
            sent1, sent2 = sent2.replace(' だ 。', ' 、'), sent1
        elif sent2.endswith(' のだ 。'):
            sent1, sent2 = sent2.replace(' のだ 。', ' ので 、'), sent1
        elif sent2.endswith(' から 。'):
            sent1, sent2 = sent2.replace(' 。', ' 、'), sent1
        return (sent1 + ' ' + sent2).replace(old, '')

    def confirm_inputs(self, example: BJWSCExample) -> None:
        inputs = self.tokenizer(*self._get_input_text(example), return_tensors='pt')
        inputs = {key: value.squeeze(0).tolist() for key, value in inputs.items()}

        input_keys = inputs.keys()
        if 'token_type_ids' not in input_keys:
            inputs['token_type_ids'] = ...
        if keys := (input_keys - {'input_ids', 'attention_mask', 'token_type_ids'}):
            print(f'unexpected keys: {keys}')

        values = [
            ' '.join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'])),
            inputs['attention_mask'],
            inputs['token_type_ids']
        ]
        print('*** confirm inputs ***')
        pprint(values, width=max(len(str(value)) for value in values) + 7)

    @staticmethod
    def _get_input_text(example: BJWSCExample) -> tuple[str]:
        return example.input,
