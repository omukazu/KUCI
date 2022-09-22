from argparse import ArgumentParser
from collections import Counter

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')

    buf = []
    for core_event_pair in df['core_event_pair']:
        core_event1, core_event2 = core_event_pair.replace('(single)', '').split('|')

        if '(complemented)' in core_event2:
            core_event2 = core_event2.split(',')[-1]  # 末尾の述語

        buf.append(core_event1)
        buf.append(core_event2)

    ctr = Counter(buf).most_common()
    with open(args.OUTPUT, mode='w') as f:
        for idx, (core_event, _) in enumerate(ctr):
            if idx < 200:
                f.write(f'{core_event}\n')


if __name__ == '__main__':
    main()
