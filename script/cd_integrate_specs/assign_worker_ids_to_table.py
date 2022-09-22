from argparse import ArgumentParser
from collections import defaultdict as ddict
from pathlib import Path

import pandas as pd
from tqdm import tqdm


tqdm.pandas()


def main():
    parser = ArgumentParser()
    parser.add_argument('TABLE', type=str, help='path to table (directory)')
    parser.add_argument('RESULT', type=str, help='path to result (directory)')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('-v', type=int, help='number of version')
    args = parser.parse_args()

    # v1.tsv (200406_arranged_cbeps.tsv)
    # v2.tsv (200501_arranged_cbeps.tsv)
    # v3.tsv (additional.tsv)
    table_dir = Path(args.TABLE)
    dfs = []
    for table_path in table_dir.iterdir():
        dfs.append(pd.read_csv(table_path, sep='\t'))
    df = pd.concat(dfs, axis=0, join='outer')

    result_dir = Path(args.RESULT)
    lines = []
    for result_path in result_dir.iterdir():
        with result_path.open(mode='r', errors='replace') as f:
            lines.extend(
                [
                    line.strip().split('\t') for line in f
                    if not (line.startswith('c') or line.startswith('d'))
                ]
            )

    num_choices = len(lines[0][3].split('@'))
    aggregated = ddict(lambda: [[] for _ in range(num_choices)])
    for line in lines:
        choices = line[3].split('@')
        normalized_event1, normalized_event2 = map(lambda x: x[3:], line[2].split('####')[1].split('##'))
        key = f'{normalized_event1}@{normalized_event2}'

        suffix = '+'
        while key in aggregated.keys() and sum(len(val) for val in aggregated[key]) >= 4:
            key = f'{key}{suffix}'
            suffix += '+'

        # 0: A is the cause/reason of B, 1: other or no relation
        aggregated[f'{key}'][choices.index(line[4])].append(line[5])

    columns = ['old_normalized_event1', 'old_normalized_event2', 'normalized_event1', 'normalized_event2']
    df['event_pair'] = df[columns].progress_apply(
        lambda x: '@'.join(x[2:4] if type(x[0]) == float or type(x[1]) == float else x[0:2]), axis=1
    )
    df['worker_ids'] = df['event_pair'].progress_apply(lambda x: aggregated[x])
    df.to_csv(args.OUTPUT, index=False, sep='\t')


if __name__ == '__main__':
    main()
