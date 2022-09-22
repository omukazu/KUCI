from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm


tqdm.pandas()


def append_negation_suffix(row: pd.Series) -> str:
    core_event_pair, negation1, negation2 = row
    core_event1, core_event2 = core_event_pair.split('|')
    core_event1 += 'n' * int(negation1)
    core_event2 += 'n' * int(negation2)
    return f'{core_event1}|{core_event2}'


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')
    df['core_event_pair'] = df[['core_event_pair', 'negation1', 'negation2']].progress_apply(
        append_negation_suffix, axis=1
    )
    df.to_csv(args.INPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()

