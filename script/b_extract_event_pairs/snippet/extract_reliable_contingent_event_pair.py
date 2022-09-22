from argparse import ArgumentParser

import pandas as pd


def contingent_and_reliable(row: pd.Series) -> bool:
    return (row['rel'] in {'原因・理由', '条件'}) and row['reliable']


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')
    df = df[df.apply(lambda x: contingent_and_reliable(x), axis=1)]
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
