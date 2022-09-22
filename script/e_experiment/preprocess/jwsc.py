import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Union

import numpy as np
from pyknp import Juman
from sklearn.model_selection import GroupKFold

from utils import tqdm


class JWSCExample:
    def __init__(self, buf: list[str]):
        self.orig, self.pronoun, antecedent_candidates, self.right_antecedent = map(
            lambda x: x.split('\t')[1], buf
        )
        candidate1, candidate2 = antecedent_candidates.replace('\u3000', '').split('、')
        self.wrong_antecedent = candidate2 if candidate1 == self.right_antecedent else candidate1
        self.label = int(candidate2 == self.right_antecedent)

        self.jumanpp = Juman()

    def dump_kwargs(self) -> tuple[dict[str, ...], list[str]]:
        morphemes = []
        for morpheme in self.jumanpp.analysis(self.orig).mrph_list():
            # Juman++ Version: 2.0.0-rc3
            if '彼は誰' == morpheme.midasi:
                morphemes.extend(['彼', 'は', '誰'])
            elif 'アンは' == morpheme.midasi:
                morphemes.extend(['アン', 'は'])
            elif '切り裂きジャック' == morpheme.midasi:
                morphemes.extend(['切り裂き', 'ジャック'])
            else:
                morphemes.append(morpheme.midasi)

        num_characters = np.array([len(morpheme) for morpheme in morphemes])
        start_indices = list(np.cumsum(num_characters) - num_characters)

        kwargs = {'orig': ' '.join(morphemes)}
        for key in ['pronoun', 'right_antecedent', 'wrong_antecedent']:
            surf = getattr(self, key)
            if key == 'pronoun':
                if self.orig.count('。') == 1:
                    start_index = start_indices.index(self.orig.find(surf))
                elif self.orig.count('。') == 2:
                    start_index = start_indices.index(
                        self.orig.find('。') + 1 + self.orig.split('。')[1].find(surf)
                    )
                else:
                    raise IndexError('the original is invalid')
            else:
                start_index = start_indices.index(self.orig.find(surf))

            num_character = num_characters[start_index]
            buf = [start_index]
            while num_character < len(surf):
                start_index += 1
                num_character += num_characters[start_index]
                buf.append(start_index)
            assert num_character == len(surf)

            kwargs[f'{key}_indices'] = buf

        return kwargs, morphemes


def replace_pronoun_with_antecedent(line: dict[str, ...], key: str) -> list[str]:
    morphemes = line['orig'].split(' ')
    morphemes[line['pronoun_indices'][0]:line['pronoun_indices'][-1] + 1] = morphemes[
        line[f'{key}_indices'][0]:line[f'{key}_indices'][-1] + 1
    ]
    return [
        ' '.join(morphemes[:line['pronoun_indices'][0]]),
        ' '.join(morphemes[line['pronoun_indices'][0]:])
    ]


def reformat(buf: list[str], format_: Literal['binary', 'span'] = 'binary') -> Union[dict, list]:
    example = JWSCExample(buf)
    kwargs, morphemes = example.dump_kwargs()
    if format_ == 'binary':
        problem = [
            {
                'input': replace_pronoun_with_antecedent(kwargs, 'right_antecedent'),
                'label': 1,
                'orig': kwargs['orig']
            },
            {
                'input': replace_pronoun_with_antecedent(kwargs, 'wrong_antecedent'),
                'label': 0,
                'orig': kwargs['orig']
            }
        ]
    elif format_ == 'span':
        label = example.label
        prefix1, prefix2 = ('right', 'wrong') if label == 0 else ('wrong', 'right')
        problem = {
            'input': kwargs['orig'],
            'pronoun_indices': kwargs['pronoun_indices'],
            'antecedent1_indices': kwargs[f'{prefix1}_antecedent_indices'],
            'antecedent2_indices': kwargs[f'{prefix2}_antecedent_indices'],
            'label': label
        }
    else:
        raise ValueError('unsupported format')

    return problem


def dump_problems(
    output_path: Path,
    problems: list[list or dict],
    format_: Literal['binary', 'span'] = 'binary'
) -> None:
    with output_path.open(mode='w') as f:
        for problem in problems:
            if format_ == 'binary':
                for line in problem:
                    json.dump(line, f, ensure_ascii=False)
                    f.write('\n')
            elif format_ == 'span':
                json.dump(problem, f, ensure_ascii=False)
                f.write('\n')
            else:
                raise ValueError('unsupported format')


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--format', default='binary', type=str, choices=['binary', 'span'], help='format')
    parser.add_argument('--test', action='store_true', help='whether to process test data or not')
    args = parser.parse_args()

    with open(args.INPUT, mode='r') as f:
        problems, buf = [], []
        for line in tqdm(f):
            line = line.strip()
            if line:
                buf.append(line)
            else:
                problem = reformat(buf, format_=args.format)
                problems.append(problem)
                buf = []

    num_folds = 5

    output_root = Path(args.OUTPUT)
    if args.test:
        output_root.mkdir(parents=True, exist_ok=True)
        dump_problems(output_root.joinpath('test.jsonl'), problems, format_=args.format)
        for k in range(num_folds):
            output_dir = output_root.joinpath(f'fold{k + 1}')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir.joinpath('test.jsonl').symlink_to(output_root.joinpath('test.jsonl'))
    else:
        k_fold = GroupKFold(n_splits=num_folds)
        if args.format == 'binary':
            groups = np.repeat(np.arange(len(problems) // 2), 2)
        else:
            groups = np.arange(len(problems))
        for k, (train_indices, dev_indices) in enumerate(k_fold.split(problems, groups=groups)):
            output_dir = output_root.joinpath(f'fold{k + 1}')
            output_dir.mkdir(parents=True, exist_ok=True)
            train_problems, dev_problems = map(
                lambda x: [problems[idx] for idx in x], [train_indices, dev_indices]
            )
            dump_problems(output_dir.joinpath('train.jsonl'), train_problems, format_=args.format)
            dump_problems(output_dir.joinpath('dev.jsonl'), dev_problems, format_=args.format)


if __name__ == '__main__':
    main()
