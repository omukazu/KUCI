from argparse import ArgumentParser
from collections import defaultdict as ddict


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to result of crowdsourcing')
    parser.add_argument('--output', default=None, type=str, help='path to output')
    args = parser.parse_args()

    with open(args.INPUT, mode='r', errors='ignore') as f:
        lines = [line.strip().split('\t') for line in f]
        lines = [line for line in lines if line[0][0] not in {'c', 'd'}]

    choices = lines[0][3].split('|')

    aggregated = ddict(lambda: [0] * len(choices))
    for line in lines:
        event1, event2 = map(lambda x: x[3:], line[2].split('####')[1].split('##'))
        aggregated[f'{event1}@{event2}'][choices.index(line[4])] += 1

    if args.output:
        with open(args.output, mode='w') as f:
            for cbep, distr in aggregated.items():
                f.write(f'{cbep}\t{distr}\n')
    else:
        a2b = sum(distr[0] >= 2 for _, distr in aggregated.items())
        print(f'{choices[0]}: {a2b} pairs')
        print(f'{choices[1]}: {len(aggregated.keys()) - a2b} pairs')


if __name__ == '__main__':
    main()
