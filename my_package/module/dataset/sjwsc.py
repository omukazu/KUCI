import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, XLMRobertaTokenizer, XLMRobertaTokenizerFast

from utils import tqdm, ObjectHook


SentencePiece = (XLMRobertaTokenizer, XLMRobertaTokenizerFast)


@dataclass
class SJWSCExample:
    example_id: int
    input: str
    pronoun_indices: list[int]
    antecedent1_indices: list[int]
    antecedent2_indices: list[int]
    label: int


class SJWSC(Dataset):
    def __init__(
        self,
        input_path: Path,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        sep: bool = False,
        debug: bool = False,
        confirm_inputs: bool = False,
        **_
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sep = sep
        self.debug = debug

        self.examples = self._load_examples(input_path, confirm_inputs=confirm_inputs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        morphemes = example.input.split(' ')
        boundary = example.pronoun_indices[0]

        inputs = self.tokenizer(
            *self._get_input_text(morphemes, boundary),
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        length = torch.sum(inputs['attention_mask']).item()

        morpheme_idx2token_idx = self._get_morpheme_idx2token_idx(morphemes, boundary)

        pronoun_map, antecedent1_map, antecedent2_map = map(
            lambda x: self._convert_indices_to_map(
                set(chain.from_iterable([morpheme_idx2token_idx[i] for i in getattr(example, x)]))
            ),
            ['pronoun_indices', 'antecedent1_indices', 'antecedent2_indices']
        )
        # first pooling
        # pronoun_map, antecedent1_map, antecedent2_map = map(
        #     lambda x: self._convert_indices_to_map(
        #         set(chain.from_iterable([morpheme_idx2token_idx[i][:1] for i in getattr(example, x)[:1]]))
        #     ),
        #     ['pronoun_indices', 'antecedent1_indices', 'antecedent2_indices']
        # )

        return {
            'example_id': example.example_id,
            'inputs': inputs,
            'length': length,
            'pronoun_map': pronoun_map,
            'antecedent1_map': antecedent1_map,
            'antecedent2_map': antecedent2_map,
            'label': example.label,
        }

    def _load_examples(self, input_path: Path, confirm_inputs: bool = False) -> list[SJWSCExample]:
        examples = []
        example_id = 0
        with input_path.open(mode='r') as f:
            for line in tqdm(f):
                example = json.loads(line, object_hook=ObjectHook)
                example = SJWSCExample(
                    example_id=example_id,
                    input=example.input,  # XLM-Rもtokenizeされたinputを使う
                    pronoun_indices=example.pronoun_indices,
                    antecedent1_indices=example.antecedent1_indices,
                    antecedent2_indices=example.antecedent2_indices,
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

    def _get_morpheme_idx2token_idx(self, morphemes: list[str], boundary: int) -> list[list[int]]:
        tokens, morpheme_idx2token_idx = [], []
        for morpheme_idx, morpheme in enumerate(morphemes):
            subwords = self.tokenizer.tokenize(morpheme)
            offset = 2 if self.sep and morpheme_idx >= boundary else 1
            morpheme_idx2token_idx.append(
                [len(tokens) + offset + shift for shift in range(len(subwords))]
            )
            tokens.extend(subwords)
        return morpheme_idx2token_idx

    def _convert_indices_to_map(self, token_indices: set[int]) -> torch.Tensor:
        return torch.as_tensor([i in token_indices for i in range(self.max_seq_len)])

    def confirm_inputs(self, example: SJWSCExample) -> None:
        morphemes = example.input.split(' ')
        boundary = example.pronoun_indices[0]

        inputs = self.tokenizer(
            *self._get_input_text(morphemes, boundary),
            return_tensors='pt'
        )
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        input_keys = inputs.keys()
        if 'token_type_ids' not in input_keys:
            inputs['token_type_ids'] = ...
        if keys := (input_keys - {'input_ids', 'attention_mask', 'token_type_ids'}):
            print(f'unexpected keys: {keys}')

        morpheme_idx2token_idx = self._get_morpheme_idx2token_idx(morphemes, boundary)

        pronoun, right_antecedent, wrong_antecedent = map(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][
                    list(chain.from_iterable([morpheme_idx2token_idx[i] for i in getattr(example, x)]))
                ].tolist()
            ),
            ['pronoun_indices', 'antecedent1_indices', 'antecedent2_indices']
        )

        values = [
            ' '.join(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'])),
            inputs['attention_mask'].tolist(),
            inputs['token_type_ids'].tolist(),
            pronoun,
            right_antecedent,
            wrong_antecedent
        ]
        print('*** confirm inputs ***')
        pprint(values, width=max(len(str(value)) for value in values) + 7)

    def _get_input_text(self, morphemes: list[str], boundary: [int]) -> tuple:
        if self.sep:
            return ' '.join(morphemes[:boundary]), ' '.join(morphemes[boundary:])
        else:
            return ' '.join(morphemes),
