import json
import re
from argparse import ArgumentParser
from typing import Literal

import pandas as pd

from utils import ObjectHook


def make_line(
    cbep: tuple[str, str],
    cfg,
    type_: Literal['c', 'd', 't'] = 't',
    id_: str = '',
    answer: str = ''
) -> str:
    normalized_event1, normalized_event2 = cbep

    if type_ in {'c', 'd'}:
        task_id = type_
    else:
        task_id = type_ + id_

    return (
        f'{task_id}\t'
        f'{int(type_ == "c")}\t'
        f'{answer}\t'
        f'{cfg.description}####A. {normalized_event1}##B. {normalized_event2}\t'
        f'{cfg.choices}\n'
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--dummy', default=None, type=str, help='path to dummy')
    parser.add_argument('--id', default='', type=str, help='identifier for analysis')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    if args.id:
        assert re.fullmatch(r'[a-zA-Z ]+', args.id) and len(args.id) < 15, 'id should be at most 14 characters in length'

    df = pd.read_csv(args.INPUT, sep='\t')
    cbeps = df[['normalized_event1', 'normalized_event2']].values

    with open(args.OUTPUT, mode='w') as f:
        f.write('設問ID(半角英数字20文字以内)\tチェック設問有無(0:無 1:有)\tチェック設問の解答(F02用)\tF01:ラベル\tF02:ラジオボタン\n')

        for check_problem in cfg.check_problems:
            answer, event1, event2 = check_problem.split('@')
            line = make_line((event1, event2), cfg, type_='c', id_='-1', answer=answer)
            f.write(line)

        for idx, cbep in enumerate(cbeps):
            line = make_line(cbep, cfg, type_='t', id_=f'{args.id}{idx // cfg.num_problems_per_task}')
            f.write(line)

        if args.dummy:
            dummy_df = pd.read_csv(args.dummy, sep='\t').sample(frac=1, random_state=0)
            dummy_cbeps = dummy_df[['normalized_event1', 'normalized_event2']].values
            for cbep in dummy_cbeps:
                line = make_line(cbep, cfg, type_='d')
                f.write(line)


if __name__ == '__main__':
    main()
