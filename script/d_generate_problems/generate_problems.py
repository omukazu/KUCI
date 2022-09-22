import json
import random
from argparse import ArgumentParser
from collections import Counter
from gzip import GzipFile
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from utils import tqdm, ObjectHook, decorator, get_logger, set_file_handlers, trace_back


L = get_logger(__name__)


@decorator(L.info, 'generation')
def generate_problems(
    input_dir: Path,
    df: pd.DataFrame,
    params
) -> tuple[list[dict[str, ...]], list[int], np.ndarray]:
    # confirm basenames are zero-padded
    bar = tqdm(input_dir.glob('*.candidate.npy.gz'))
    morphemes1, normalized_morphemes2, core_event_pair, agreement = map(
        lambda x: df[x].values,
        ['morphemes1', 'normalized_morphemes2', 'core_event_pair', 'agreement']
    )
    problems, num_candidates, start, watcher = [], [], 0, np.zeros(len(morphemes1), dtype=np.int16)

    for input_path in bar:
        cache = np.load(GzipFile(input_path, mode='r'), allow_pickle=True)
        batch_size, _ = cache.shape
        for idx, row in enumerate(cache):
            if np.sum(row) < params.num_distractors:
                continue

            # unique has already been sorted
            unique, counts = np.unique(watcher[row], return_counts=True)
            threshold = unique[np.argmax(np.cumsum(counts) >= params.num_distractors)]
            if threshold >= params.upper_bound:
                continue

            context, correct_choice = morphemes1[start + idx], normalized_morphemes2[start + idx]
            indices, *_ = np.where(row & (watcher <= threshold))
            # indices, *_ = np.where(row)  # not consider the frequency of reuse
            redundant = normalized_morphemes2[indices]  # contain duplicates
            candidates, uniq_indices = np.unique(redundant, return_index=True)
            num_candidate = len(candidates)
            num_candidates.append(num_candidate)
            if num_candidate < params.num_distractors:
                continue

            choice_indices = np.random.choice(uniq_indices, params.num_distractors, replace=False)
            choices = redundant[choice_indices].tolist()
            answer_idx = random.randint(0, params.num_distractors)
            choices.insert(answer_idx, correct_choice)

            problem = {
                'id': len(problems),
                'context': context,
                **{f'choice_{chr(idx + 97)}': cho for idx, cho in enumerate(choices)},
                'label': chr(answer_idx + 97),
                'agreement': int(agreement[start + idx]),
                'core_event_pair': core_event_pair[start + idx],
            }
            # another format
            # choices = '\n'.join(f'{idx + 1}.{choice}' for idx, choice in enumerate(choices))
            # problem = f'{ctx}\n{choices}\nanswer:{ans_idx + 1}\n\n'
            problems.append(problem)

            watcher[indices[choice_indices]] += 1
            bar.set_postfix({
                'num_problems': len(problems),
                'num_candidate': num_candidate,
                'threshold': threshold,
                'max': watcher.max()
            })
        start += batch_size
    return problems, num_candidates, watcher


def show_statistics(num_candidates: list[int], watcher: np.array) -> None:
    sorted_num_candidates, length = sorted(num_candidates), len(num_candidates)
    mean = floor(sum(num_candidates) / length)
    if len(num_candidates) % 2 == 0:
        median = (sorted_num_candidates[length // 2 - 1] + sorted_num_candidates[length // 2]) / 2
    else:
        median = sorted_num_candidates[length // 2]
    L.info(f'mean: {mean}')
    L.info(f'median: {median}')
    L.info(f'distribution: {json.dumps(sorted(Counter(watcher.tolist()).items()))}')


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('INPUT', type=str, help='path to input (directory)')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    random.seed(cfg.params.seed)
    np.random.seed(cfg.params.seed)

    set_file_handlers(L, output_path=Path(args.OUTPUT).with_suffix('.log'))

    try:
        df = pd.read_csv(
            cfg.path.table,
            sep='\t',
            usecols=['morphemes1', 'normalized_morphemes2', 'core_event_pair', 'worker_ids']
        )
        df['agreement'] = df['worker_ids'].map(lambda x: len(x[0]))
        df.drop(['worker_ids'], axis=1, inplace=True)
    except Exception as e:
        trace_back(e, print, 'reload df')
        df = pd.read_csv(
            cfg.path.table,
            sep='\t',
            usecols=['morphemes1', 'normalized_morphemes2', 'core_event_pair']
        )
        df['agreement'] = -1

    params = json.loads(
        json.dumps({
            'num_distractors': cfg.params.num_choices - 1,
            'upper_bound': cfg.params.num_choices
        }),
        object_hook=ObjectHook
    )
    L.info(f'params: {json.dumps(params, indent=2)}')

    problems, num_candidates, watcher = generate_problems(Path(args.INPUT), df, params)
    show_statistics(num_candidates, watcher)

    with open(args.OUTPUT, mode='w') as f:
        for problem in problems:
            json.dump(problem, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    main()
