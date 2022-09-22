import json
from argparse import ArgumentParser

from pyknp import Juman


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    with open(args.INPUT, mode='r') as f:
        problems = [json.loads(line) for line in f]

    jumanpp = Juman()

    with open(args.OUTPUT, mode='w') as f:
        for problem in problems:
            for key in problem.keys():
                if key == 'question' or key.startswith('choice'):
                    problem[key] = ' '.join(morpheme.midasi for morpheme in jumanpp.analysis(problem[key]).mrph_list())
            json.dump(problem, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    main()
