import json
from argparse import ArgumentParser
from math import ceil
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output dir')
    parser.add_argument('--ext', default=None, type=str, help='extension')
    parser.add_argument('--num-splits', default=1000, type=int, help='number of splits')
    args = parser.parse_args()

    if args.ext:
        with open(args.INPUT, mode='r') as f:
            problems = [json.loads(line) for line in f]

        chunk_size = ceil(len(problems) / args.num_splits)
        for idx in range(args.num_splits):
            with Path(args.OUTPUT).joinpath(f'{idx + 1:08}.problems.jsonl').open(mode='w') as f:
                for problem in problems[chunk_size * idx: chunk_size * (idx + 1)]:
                    json.dump(problem, f, ensure_ascii=False)
                    f.write('\n')
    else:
        problems = []
        # confirm basenames are zero-padded
        for input_path in Path(args.INPUT).glob('*.problems.jsonl'):
            with input_path.open(mode='r') as f:
                problems.extend([json.loads(line) for line in f])

        with open(args.OUTPUT, mode='w') as f:
            for problem in problems:
                json.dump(problem, f, ensure_ascii=False)
                f.write('\n')


if __name__ == '__main__':
    main()
