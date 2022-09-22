from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


def contingent_reliable_and_basic(row: pd.Series) -> bool:
    return (row['rel'] in {'原因・理由', '条件'}) and row['reliable'] and row['basic']


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')
    if len(df) > 0:
        causal, conditional = map(lambda x: df['rel'] == x, ['原因・理由', '条件'])
        contingent = causal | conditional
        reliable_contingent = contingent & df['reliable']
        df = df[
            df[['rel', 'reliable', 'basic']].apply(lambda x: contingent_reliable_and_basic(x), axis=1)
        ]
        counts = [contingent.sum(), reliable_contingent.sum(), len(df)]
    else:
        counts = [0, 0, 0]

    output_path = Path(args.OUTPUT)
    df.to_csv(output_path, sep='\t', index=False)
    with open(output_path.with_suffix(f'{"".join(output_path.suffixes)}.count'), mode='w') as f:
        f.write('\t'.join(list(map(str, counts))))


if __name__ == '__main__':
    main()
