import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast

from utils import tqdm, ObjectHook


# Necessary to input unsegmented sentences when using the SentencePiece-based tokenizers
SentencePiece = (XLMRobertaTokenizer, XLMRobertaTokenizerFast)


@dataclass
class KUCIExample:
    example_id: int
    context: str
    choices: list[str]
    label: int


class KUCI(Dataset):
    def __init__(
        self,
        input_path: Path,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        weight: float = 1.,
        num_examples: int = None,
        choice_only: bool = False,
        debug: bool = False,
        confirm_inputs: bool = False,
        **_
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.weight = weight
        self.num_examples = num_examples
        self.choice_only = choice_only
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
            return_length=True
        )
        length = inputs.pop('length')
        # length = torch.max(torch.sum(inputs['attention_mask'], axis=1)).item()

        return {
            'example_id': example.example_id,
            'inputs': inputs,
            'length': length,
            'weight': self.weight,
            'label': example.label,
        }

    def _load_examples(self, input_path: Path, confirm_inputs: bool = False) -> list[KUCIExample]:
        examples = []
        example_id = 0
        with input_path.open(mode='r') as f:
            for line in tqdm(f):
                if isinstance(self.num_examples, int) and len(examples) == self.num_examples:
                    break
                elif self.debug and len(examples) == 10:
                    break

                example = json.loads(line, object_hook=ObjectHook)
                old = ' ' if isinstance(self.tokenizer, SentencePiece) else ''
                example = KUCIExample(
                    example_id=example_id,
                    context=example.context.replace(old, ''),
                    choices=[val.replace(old, '') for key, val in example.items() if key.startswith('choice')],
                    label=ord(example.label) - 97
                )
                examples.append(example)
                example_id += 1

        if confirm_inputs:
            print('*** confirm inputs ***')
            for i in range(1, 2):
                self.confirm_inputs(examples[-i])

        return examples

    def confirm_inputs(self, example: KUCIExample) -> None:
        inputs = self.tokenizer(
            *self._get_input_text(example),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )  # (num_choices, max_seq_len)
        inputs = {key: value.tolist() for key, value in inputs.items()}

        input_keys = inputs.keys()
        if 'token_type_ids' not in input_keys:
            inputs['token_type_ids'] = ...
        if keys := (input_keys - {'input_ids', 'attention_mask', 'token_type_ids'}):
            print(f'unexpected keys: {keys}')

        values = [
            [
                ' '.join(
                    map(
                        lambda x: '-' if x == self.tokenizer.pad_token else x,
                        self.tokenizer.convert_ids_to_tokens(input_ids)
                    )
                ),
                attention_mask,
                token_type_ids
            ]
            for input_ids, attention_mask, token_type_ids in zip(*inputs.values())
        ]
        print('*** confirm inputs ***')
        pprint(values, width=max(max(map(lambda x: len(str(x)), inputs)) for inputs in values) + 7)

    def _get_input_text(self, example: KUCIExample) -> tuple[list]:
        if self.choice_only:
            return example.choices,
        else:
            return [[example.context, choice] for choice in example.choices],
