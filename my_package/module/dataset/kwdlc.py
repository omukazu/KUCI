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
class KWDLCExample:
    example_id: int
    clause1: str
    clause2: str
    label: int


class KWDLC(Dataset):
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

        self.idx2cls = ['談話関係なし', '原因・理由', '条件', '目的', '根拠', '対比', '逆接']

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
            return_tensors='pt'
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        length = torch.sum(inputs['attention_mask']).item()

        return {
            'example_id': example.example_id,
            'inputs': inputs,
            'length': length,
            'label': example.label,
        }

    def _load_examples(self, input_path: Path, confirm_inputs: bool = False) -> list[KWDLCExample]:
        examples = []
        example_id = 0
        with input_path.open(mode='r') as f:
            for line in tqdm(f):
                example = json.loads(line, object_hook=ObjectHook)
                old = ' ' if isinstance(self.tokenizer, SentencePiece) else ''
                example = KWDLCExample(
                    example_id=example_id,
                    clause1=example.clauses[example['clause1_index']].replace(old, ''),
                    clause2=example.clauses[example['clause2_index']].replace(old, ''),
                    label=self.idx2cls.index(example.sense)
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

    def confirm_inputs(self, example: KWDLCExample) -> None:
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
    def _get_input_text(example: KWDLCExample) -> tuple[str, str]:
        return example.clause1, example.clause2
