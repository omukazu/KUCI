from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import regex

from utils import get_logger, set_file_handlers


WHITELIST = regex.compile(r'[\p{Hiragana}\p{Katakana}\p{Han}0-9a-zA-Z０-９ａ-ｚＡ-Ｚー。、()（）「」]+')
L = get_logger(__name__)


def is_trivial_core_event_pair(row: pd.Series, blacklist: set[str]) -> bool:
    core_event1, core_event2 = row['core_event_pair'].replace('(single)', '').split('|')
    if '(complemented)' in core_event2:
        core_event2 = core_event2.lstrip('(complemented)')
    return (core_event1 in blacklist) or (core_event2 in blacklist) or (core_event1 == core_event2)


def filter_by_rules(row: pd.Series, blacklist: set[str]) -> list[bool]:
    """Filter by rules

    NOTE:
        trivial_core_event_pair: whether event pair contains high-frequency/identical core event pair or not
        multiple_base_phrases: whether number of base phrases in latter event is more than one or not
        noun_phrases: whether type of former or latter event is noun or not
        pronoun_or_undefined: whether former or latter event contains demonstratives or undefined words
        contain_special_symbols: " special symbols
        not_closed_brackets: whether brackets in former and latter events are closed or not
        rel_surf: whether former and latter events are connected with discourse marker "なら" or not
    """

    if row.name % 10000 == 0:
        print(f'*** filtering row No.{row.name} ***')

    event1, event2 = row['event1'], row['normalized_event2']

    trivial_core_event_pair = is_trivial_core_event_pair(row, blacklist)
    multiple_base_phrases = (row['num_base_phrases2'] > 1)
    noun_phrase = any(row[key] == '判' for key in ['type1', 'type2'])
    pronoun_or_undefined = any(row[key] for key in ['pou1', 'pou2'])
    contain_special_symbols = any(WHITELIST.fullmatch(surf) is None for surf in [event1, event2])
    closed_bracket = all(surf.count('「') == surf.count('」') for surf in [event1, event2])
    rel_surf = (row['rel_surf'] != '〜なら')

    return [
        not trivial_core_event_pair,
        multiple_base_phrases,
        not noun_phrase,
        not pronoun_or_undefined,
        not contain_special_symbols,
        closed_bracket,
        rel_surf,
    ]


def aggregate_pas_pair(group: pd.DataFrame) -> int:
    num_characters = list(group[['event1', 'normalized_event2']].apply(lambda x: len(x[0] + x[1]), axis=1))
    return num_characters.index(max(num_characters))


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--blacklist', type=str, help='path to blacklist of core events')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')

    output_path = Path(args.OUTPUT)
    set_file_handlers(L, output_path=output_path.parent.joinpath('post_process.log'))

    with open(args.blacklist, mode='r') as f:
        blacklist = {line.strip() for line in f if not line.startswith('#')}

    df['rules'] = df.apply(filter_by_rules, blacklist=blacklist, axis=1)
    L.debug(f'total: {len(df)} pairs')

    keys = [
        'trivial core event pair',
        'multiple base phrases',
        'noun phrases',
        'pronoun or undefined',
        'contain special symbols'
        'closed brackets',
        'rel surf'
    ]
    values = np.sum(np.array(df['rules'].values.tolist()), axis=0)
    for key, val in zip(keys, values):
        L.debug(f'{key}: {val} pairs')
    df = df[df['rules'].map(lambda x: all(x))]

    # de-duplicate PAS, keeping the event pair with the most characters
    df = df.groupby(['pas1', 'pas2'], as_index=False).apply(lambda x: x.iloc[aggregate_pas_pair(x)])
    # de-duplicate event pairs, keeping the first one
    df = df.groupby(['normalized_event1', 'normalized_event2'], as_index=False).apply(lambda x: x.iloc[0])
    df.drop(['pou1', 'pou2', 'rules'], axis=1, inplace=True)
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
