from argparse import ArgumentParser

import pandas as pd

from utils import tqdm

tqdm.pandas()


def aggregate_pas_pair(group: pd.DataFrame) -> int:
    num_characters = list(group[['event1', 'normalized_event2']].apply(lambda x: len(x[0] + x[1]), axis=1))
    return num_characters.index(max(num_characters))


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--table_w_worker_ids', type=str, help='path to table_w_worker_ids.tsv')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')

    table_w_worker_ids = pd.read_csv(
        args.table_w_worker_ids,
        sep='\t',
        usecols=['orig', 'worker_ids', 'core_event_pair']
    )
    converter = {
        orig: {
            'worker_ids': eval(worker_ids),
            'core_event_pair': core_event_pair
        }
        for orig, worker_ids, core_event_pair in table_w_worker_ids.values
    }

    verified = set(table_w_worker_ids['orig'].values)
    df = df[
        [
            orig in verified and rel in {'原因・理由', '条件'} and reliable and num_bps2 > 1
            for orig, rel, reliable, num_bps2 in df[['orig', 'rel', 'reliable', 'num_bps2']].values
        ]
    ]

    df['worker_ids'] = df['orig'].progress_apply(lambda x: converter[x]['worker_ids'])
    df = df[df['worker_ids'].progress_apply(lambda x: len(x[0]) >= 2)]
    df['core_event_pair'] = df['orig'].progress_apply(lambda x: converter[x]['core_event_pair'])
    df = df.groupby(['pas1', 'pas2'], as_index=False).progress_apply(lambda x: x.iloc[aggregate_pas_pair(x)])
    df = df.groupby(['normalized_event1', 'normalized_event2'], as_index=False).progress_apply(lambda x: x.iloc[0])
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
