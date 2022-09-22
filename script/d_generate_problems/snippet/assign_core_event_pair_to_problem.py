import json
import os
from argparse import ArgumentParser

import pandas as pd

from utils import ObjectHook


def append_negation_suffix(core_event_pair: str, negation1: bool, negation2: bool) -> str:
    core_event1, core_event2 = core_event_pair.split('|')
    core_event1 += 'n' * int(negation1)
    core_event2 += 'n' * int(negation2)
    return f'{core_event1}|{core_event2}'


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    df = pd.read_csv(
        cfg.table,
        sep='\t',
        usecols=['event1', 'normalized_event2', 'negation1', 'negation2', 'core_event_pair']
    )
    cbep2core_event_pair = {
        f'{event1}{normalized_event2}': append_negation_suffix(core_event_pair, negation1, negation2)
        for event1, normalized_event2, negation1, negation2, core_event_pair in df.values
    }

    for stem in ['train', 'dev', 'test']:
        with open(os.path.join(cfg.dataset, f'{stem}.jsonl'), mode='r') as f:
            problems = [json.loads(line) for line in f]

        with open(os.path.join(cfg.dataset, f'{stem}.jsonl'), mode='w') as f:
            for problem in problems:
                context, correct_choice = map(
                    lambda x: problem[x].replace(' ', ''),
                    ['context', f'choice_{problem.label}']
                )
                core_event_pair = cbep2core_event_pair[f'{context}{correct_choice}']
                problem['core_event_pair'] = core_event_pair
                json.dump(problem, f, ensure_ascii=False)
                f.write('\n')


if __name__ == '__main__':
    main()

