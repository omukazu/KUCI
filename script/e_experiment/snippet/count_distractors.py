import json
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from pathlib import Path
from pprint import pprint

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--table', type=str, help='path to table.tsv')
    args = parser.parse_args()

    input_dir = Path(args.INPUT)
    split = []
    for stem in ['train', 'dev', 'test']:
        with open(input_dir.joinpath(f'{stem}.jsonl'), mode='r') as f:
            split.append([json.loads(line) for line in f])

    df = pd.read_csv(args.table, sep='\t')

    num_choices = sum(key.startswith('choice') for key in split[0][0].keys())

    distractors = []
    for problem in chain.from_iterable(split):
        for idx in range(num_choices):
            if problem['label'] != chr(idx + 97):
                distractors.append(problem[f'choice_{chr(idx + 97)}'].replace(' ', ''))

    distractor_ctr = Counter(distractors)
    frequency_ctr = Counter(distractor_ctr.values())
    unused_count = len(set(df['normalized_event2'].values) - set(distractor_ctr.keys()))

    with open(args.OUTPUT, mode='w') as f:
        f.write(f'0\t{unused_count}\n')
        for frequency, count in sorted(frequency_ctr.items(), key=lambda x: x[0]):
            f.write(f'{frequency}\t{count}\n')


if __name__ == '__main__':
    main()
