from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--ext', default=None, type=str, help='extension')
    parser.add_argument('--num-splits', default=1000, type=int, help='number of splits')
    args = parser.parse_args()

    if args.ext:
        df = pd.read_csv(args.INPUT, sep='\t')

        chunk_size = ceil(len(df) / args.num_splits)

        output_dir = Path(args.OUTPUT)
        for idx in range(args.num_splits):
            chunk = df.iloc[chunk_size * idx: chunk_size * (idx + 1)]
            if len(chunk) > 0:
                chunk.to_csv(output_dir.joinpath(f'{idx:08}{args.ext}'), sep='\t', index=False)
    else:
        dfs = [pd.read_csv(input_path, sep='\t') for input_path in Path(args.INPUT).glob('*')]
        pd.concat(dfs).to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
