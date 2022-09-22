import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--aggregated', type=str, help='path to aggregated.tsv')
    args = parser.parse_args()

    with open(args.aggregated, mode='r') as f:
        lines = [line.strip().split('\t') for line in f]
    cbep2distr = {cbep: eval(distr) for cbep, distr in lines}

    df = pd.read_csv(args.INPUT, sep='\t')
    df['distr'] = df[['normalized_event1', 'normalized_event2']].apply(
        lambda x: cbep2distr[f'{x[0]}@{x[1]}'], axis=1
    )
    df = df[df['distr'].map(lambda x: x[0] >= 2)]
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
