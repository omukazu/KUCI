import json
import random
from argparse import ArgumentParser
from pathlib import Path

from sklearn.model_selection import GroupKFold

from utils import ObjectHook, decorator, get_logger, set_file_handlers


L = get_logger(__name__)


@decorator(L.info, 'loading')
def load_problems(input_path: Path) -> tuple[list[dict[str, ...]], list[str]]:
    with open(input_path, mode='r') as f:
        problems = [json.loads(line) for line in f]
    random.shuffle(problems)
    core_event_pairs = [trim_core_event_pair(problem['core_event_pair']) for problem in problems]
    return problems, core_event_pairs


def trim_core_event_pair(core_event_pair: str) -> str:
    buf = []
    for core_event in core_event_pair.split('|'):
        if '(complemented)' in core_event:
            buf.append(core_event.split(',')[-1])
        else:
            buf.append(core_event)
    return '|'.join(buf)


@decorator(L.info, 'applying GroupKFold')
def apply_group_k_fold(problems: list[dict], core_eps: list[str]) -> list[tuple]:
    return list(GroupKFold(n_splits=10).split(problems, groups=core_eps))


@decorator(L.info, 'splitting problems')
def split_problems(
    problems: list[dict[str, ...]],
    core_event_pairs: list[str]
) -> tuple[list[dict[str, ...]], list[dict[str, ...]], list[dict[str, ...]]]:
    gkf = apply_group_k_fold(problems, core_event_pairs)

    dev_indices, test_indices = gkf[-2][1], gkf[-1][1]
    assert len(set(dev_indices) & set(test_indices)) == 0, 'there are some duplicates between dev and test'
    train_indices = sorted(list(set(range(len(problems))) - (set(dev_indices) | set(test_indices))))

    train, dev, test = map(
        lambda x: [problems[idx] for idx in x], [train_indices, dev_indices, test_indices]
    )

    train_context_and_choice_pairs = {
        f'{problem["context"]}@{problem[key]}'
        for problem in train
        for key in problem.keys() if key.startswith('choice')
    }
    dev = remove_duplicates(dev, train_context_and_choice_pairs)
    test = remove_duplicates(test, train_context_and_choice_pairs)
    L.debug(f'train: {len(train_indices)}')
    L.debug(f'dev: {len(dev_indices)} -> {len(dev)}')
    L.debug(f'test: {len(test_indices)} -> {len(test)}')
    return train, dev, test


# remove some problems so that there are no duplicate context and distractor pairs between train and dev/test
def remove_duplicates(
    split: list[dict[str, ...]],
    train_context_and_choice_pairs: set[str]
) -> list[dict[str, ...]]:
    removed = []
    for problem in split:
        context_and_distractor_pairs = [
            f'{problem["context"]}@{problem[key]}'
            for key in problem.keys() if key.startswith('choice') and not key.endswith(problem['label'])
        ]
        # there have already been no duplicate context and correct choice pair
        # cf. b_extract_event_pairs/post_process.py l.99
        if any(
            context_and_distractor_pair in train_context_and_choice_pairs
            for context_and_distractor_pair in context_and_distractor_pairs
        ):
            continue
        else:
            removed.append(problem)
    return removed


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output dir')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    random.seed(cfg.params.seed)

    output_dir = Path(args.OUTPUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_file_handlers(L, output_path=output_dir.joinpath('build_dataset.log'))

    problems, core_event_pairs = load_problems(args.INPUT)
    train, dev, test = split_problems(problems, core_event_pairs)

    problem_id = 0
    for split, stem in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
        with output_dir.joinpath(f'{stem}.jsonl').open(mode='w') as f:
            for problem in split:
                json.dump({'problem_id': problem_id, **problem}, f, ensure_ascii=False)
                f.write('\n')
                problem_id += 1


if __name__ == '__main__':
    main()
