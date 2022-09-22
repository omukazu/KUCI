import json
import re
from argparse import ArgumentParser
from typing import Literal

from utils import ObjectHook


def get_context_and_choices(problem: dict[str, str]) -> tuple[str, tuple]:
    context = ''.join(problem['context'].split(' '))
    choices = tuple(
        map(lambda x: ''.join(problem[x].split(' ')), [key for key in problem.keys() if key.startswith('choice')])
    )
    return context, choices


def make_line(
    problem: tuple[str, tuple],
    description: str,
    type_: Literal['c', 'd', 't'] = 't',
    id_: str = '',
    answer: str = ''
) -> str:
    context, choices = problem

    if type_ in {'c', 'd'}:
        task_id = type_
    else:
        task_id = type_ + id_

    return (
        f'{task_id}\t'
        f'{int(type_ == "c")}\t'
        f'{answer}\t'
        f'{description}####{context}\t'
        f'{"@".join(choices)}\n'
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('INPUT', type=str, help='path to input')  # jsonl
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--id', default='', type=str, help='identifier for analysis')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    if args.id:
        assert re.fullmatch(r'[a-zA-Z ]+', args.id) and len(args.id) < 15, 'id should be at most 14 characters in length'

    with open(args.INPUT, mode='r') as f:
        problems = [get_context_and_choices(json.loads(line)) for line in f]

    with open(args.OUTPUT, mode='w') as f:
        f.write(
            '設問ID(半角英数字20文字以内)\t'
            'チェック設問有無(0:無 1:有)\t'
            'チェック設問の解答(F02用)\t'
            'F01:ラベル\t'
            'F02:ラジオボタン\n'
        )

        for check_problem in cfg.check_problems:
            answer_idx, context, *choices = check_problem.split('@')
            line = make_line(
                (context, choices),
                cfg.description,
                type_='c',
                id_='-1',
                answer=choices[int(answer_idx) - 1]
            )
            f.write(line)

        for idx, problem in enumerate(problems):
            line = make_line(
                problem,
                cfg.description,
                type_='t',
                id_=f'{args.id}{idx // cfg.num_problems_per_task}'
            )
            f.write(line)


if __name__ == '__main__':
    main()
