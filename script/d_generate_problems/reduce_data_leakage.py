import gc
import json
import os
from argparse import ArgumentParser
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ObjectHook, decorator, get_logger, set_file_handlers, multi_processing


tqdm.pandas()
L = get_logger(__name__)


@decorator(L.info, 'loading')
def load_data(cfg) -> tuple[pd.DataFrame, list[list[str]], set[str], set[str], dict[str, int]]:
    df = pd.read_csv(cfg.table, sep='\t')
    df = df.assign(
        idx=df.index.values,
        segmented_cbep=df[['morphemes1', 'normalized_morphemes2']].progress_apply(lambda x: ' '.join(x), axis=1)
    )

    eval_problems = []
    for split in ['dev', 'test']:
        with open(os.path.join(cfg.dataset, f'{split}.jsonl'), mode='r') as f:
            eval_problems += [json.loads(line) for line in f]

    eval_cbeps, eval_core_event_pairs, vocab = set(), set(), set()
    for eval_problem in eval_problems:
        context, correct_choice = map(
            lambda x: eval_problem[x].split(' '), ['context', f'choice_{eval_problem["label"]}']
        )
        eval_cbeps.add(''.join(context + correct_choice))
        eval_core_event_pairs.add(eval_problem['core_event_pair'])
        vocab |= set(context + correct_choice)
        eval_problem = context + correct_choice
    vocab |= set(chain.from_iterable(map(lambda x: x.split(' '), df['segmented_cbep'])))
    encoder = {word: idx for idx, word in enumerate(vocab)}

    eval_problems = [
        np.array([encoder[word] for word in words], dtype='int32') for words in eval_problems
    ]

    return df, eval_problems, eval_cbeps, eval_core_event_pairs, encoder


def compute_lcs(query: np.array, table: np.array) -> np.array:
    query_seq_len, = query.shape
    num_rows, max_seq_len = table.shape

    query, table = query[None, :, None], table[:, None, :]  # (1, src_seq_len, 1), (batch_size, 1, max_seq_len)
    equal = (query == table)   # batch_size, src_seq_len, max_seq_len

    # dp1: longest common substring, dp2: longest common subsequence
    # dp1 = np.zeros((batch_size, query_seq_len + 1, max_seq_len + 1), dtype=int)
    dp2 = np.zeros((num_rows, query_seq_len + 1, max_seq_len + 1), dtype='int32')
    # https://qiita.com/yH3PO4/items/332c1ee51c5131032b8e#f---lcs
    for idx in tqdm(range(query_seq_len), leave=False):
        # dp1[:, idx + 1, 1:] = (dp1[:, idx, :-1] + equal[:, idx]) * equal[:, idx]
        dp2[:, idx + 1, 1:] = np.maximum(dp2[:, idx, 1:], dp2[:, idx, :-1] + equal[:, idx])
        dp2[:, idx + 1] = np.maximum.accumulate(dp2[:, idx + 1], axis=1)

    del equal
    gc.collect()

    return dp2


def find_high_lexical_overlap(
    query: np.array,
    table: np.array,
    threshold: float = 0.75
) -> tuple[np.array, list[list[int]]]:
    dp = compute_lcs(query, table)
    query_seq_len, = query.shape
    indices, buf = np.where(dp[:, -1, -1] >= query_seq_len * threshold)[0], []
    # indices, buf = np.where(dp[:, -1, -1] >= int(key_seq_len * threshold))[0], []
    for idx in indices:
        # i, j = np.unravel_index(np.argmax(dp[idx]), dp[idx].shape)
        # n_gram_overlap = key[idx, i - dp[idx][i][j]: i]

        row, mtx = map(lambda x: x[idx], [table, dp])
        i, j = query_seq_len, np.sum(row != -1)

        lcs = []
        while i > 0 and j > 0:
            if query[i - 1] == row[j - 1]:
                lcs.append(row[i - 1])
                i -= 1
                j -= 1
            elif mtx[i][j] == mtx[i - 1][j]:
                i -= 1
            elif mtx[i][j] == mtx[i][j - 1]:
                j -= 1
        buf.append(lcs[::-1])

    return indices, buf


def get_data_leakage(eval_problems: list[list[str]], encoder: dict[str, int], table: np.array) -> set[int]:
    decoder = {id_: word for word, id_ in encoder.items()}

    data_leakage, count = set(), 0
    for query in tqdm(eval_problems, leave=False):
        indices, buf = find_high_lexical_overlap(query, table)
        data_leakage |= set(indices.tolist())
        if len(indices) > 0 and count < 10:
            example = {
                'eval_cbep': ' '.join(decoder[id_] for id_ in query),
                'data_leakage': ' '.join(decoder[id_] for id_ in table[indices[0]] if id_ >= 0),
                'lcs': f'{" ".join(decoder[id_] for id_ in buf[0])} ({len(buf[0])} / {len(query)})'
            }
            L.info(f'{json.dumps(example, ensure_ascii=False, indent=2)}')
            count += 1

    return data_leakage


def reduce_data_leakage(
    row: pd.Series,
    data_leakage: set[int],
    eval_cbeps: set[str],
    eval_core_event_pairs: set[str]
) -> bool:
    idx, event1, normalized_event2, core_event_pair = row
    return \
        idx not in data_leakage and \
        f'{event1}{normalized_event2}' not in eval_cbeps and \
        core_event_pair not in eval_core_event_pairs


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('-j', default=1, type=int, help='number of jobs')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    set_file_handlers(L)

    df, eval_problems, eval_cbeps, eval_core_event_pairs, encoder = load_data(cfg)

    table = [
        [encoder[word] for word in segmented_cbep.split(' ')] for segmented_cbep in df['segmented_cbep']
    ]
    max_seq_len = len(max(table, key=len))
    table = np.array([row + [-1] * (max_seq_len - len(row)) for row in table], dtype='int32')

    chunks = multi_processing(eval_problems, get_data_leakage, args=(encoder, table), num_jobs=args.j)
    data_leakage = set(chain.from_iterable(map(lambda x: x, chunks)))

    before = len(df)
    df = df[
        df[['idx', 'event1', 'normalized_event2', 'core_event_pair']].progress_apply(
            lambda x: reduce_data_leakage(x, data_leakage, eval_cbeps, eval_core_event_pairs), axis=1
        )
    ]
    after = len(df)
    df.drop(['idx'], axis=1, inplace=True)
    df.to_csv(args.OUTPUT, sep='\t', index=False)
    print(f'reduced: {before} -> {after}')


if __name__ == '__main__':
    main()
