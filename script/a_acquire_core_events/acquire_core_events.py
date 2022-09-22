import json
from argparse import ArgumentParser
from collections import defaultdict as ddict
from itertools import product
from pathlib import Path
from time import sleep
from typing import Generator

import numpy as np

from utils import tqdm, ObjectHook, decorator, get_logger, set_file_handlers, trace_back


TRANSLATION_TABLE = {
    'ノ格~判': 'の/の',
    'ノ格~ガ格': 'の/の', 'ガ格': 'ガ',
    'ノ格~ヲ格': 'の/の', 'ヲ格': 'ヲ',
    'ノ格~ニ格': 'の/の', 'ニ格': 'ニ',
    'ノ格~デ格': 'の/の', 'デ格': 'デ',
    'ノ格~ト格': 'の/の', 'ト格': 'ト',
    'ノ格~ヘ格': 'の/の', 'ヘ格': 'ヘ',
    'ノ格~カラ格': 'の/の', 'カラ格': 'カラ',
    'ノ格~マデ格': 'の/の', 'マデ格': 'マデ',
    'ノ格~ヨリ格': 'の/の', 'ヨリ格': 'ヨリ',
    '修飾': '修飾', '時間': '時間',
    'にくらべて': 'ニクラベル', 'にとって': 'ニトル', 'によって': 'ニヨル'
}
POSSESSIVE_CASES = {key for key in TRANSLATION_TABLE.keys() if key.startswith('ノ格~') and key.endswith('格')}
L = get_logger(__name__)


@decorator(L.info, 'loading')
def load_case_frame_dict(input_path: str) -> dict[str, ...]:
    with open(input_path, mode='r') as f:
        case_frame_dict = json.load(f)
    return case_frame_dict


def prune_case_frames(case_frame2cases: dict[str, ...], case_frame_rate: float) -> list[str]:
    # e.g. ['壊す/こわす:動1', '壊す/こわす:動2', ... , 'predicate_total']
    case_frames = np.array(list(case_frame2cases.keys())[:-1])  # exclude the last element (predicate_total)
    case_frame_totals = np.array([case_frame2cases[case_frame]['case_frame_total'] for case_frame in case_frames])  # (n, 1)
    sorted_indexes = np.argsort(case_frame_totals)[::-1]
    cumsum = np.cumsum(np.sort(case_frame_totals)[::-1])
    # compute threshold for selecting top-k case frames
    threshold = np.sum((cumsum / case_frame2cases['predicate_total']) < case_frame_rate) + 1
    return list(case_frames[sorted_indexes[:threshold]])


def prune_cases(case2arguments: dict[str, ...], case_rate: float) -> list[str]:
    # e.g. ['ガ格', 'ヲ格', ... , '修飾']
    cases = np.array([case for case in list(case2arguments.keys())[:-1] if case not in POSSESSIVE_CASES])
    case_totals = np.array([case2arguments[case]['case_total'] for case in cases])
    sorted_indexes = np.argsort(case_totals)[::-1]
    cumsum = np.cumsum(np.sort(case_totals)[::-1])
    threshold = np.sum((cumsum / case2arguments['case_frame_total']) < case_rate) + 1
    return list(cases[sorted_indexes[:threshold]])


def prune_arguments(argument2frequency: dict[str, float], argument_rate: float) -> tuple[list[str], float]:
    arguments = list(argument2frequency.items())[:-1]  # arguments have been sorted by frequency
    cumsum = np.cumsum(np.array([frequency for _, frequency in arguments]))
    threshold = np.sum((cumsum / argument2frequency['case_total']) < argument_rate) + 1
    return arguments[:threshold], cumsum[:threshold][-1]


@decorator(L.info, 'pruning')
def prune(
    case_frame_dict: dict[str, ...],
    cfg,
    request: bool = False
) -> tuple[ddict, set, ...]:
    pruned, intransitive_predicates = ddict(lambda: ddict(dict)), set()
    ctr = json.loads(
        json.dumps({'argument_frequency': 0, 'predicate': 0, 'predicate_frequency': 0}),
        object_hook=ObjectHook
    )

    predicate2frequency = {
        predicate: case_frame2cases['predicate_total'] for predicate, case_frame2cases in case_frame_dict.items()
    }
    for predicate, frequency in tqdm(sorted(predicate2frequency.items(), key=lambda x: x[1], reverse=True)):
        if '+' not in predicate:
            ctr.predicate_frequency += frequency

        # limit to top-k frequent single-word predicates in the active voice
        if ctr.predicate >= cfg.params.predicate_threshold or ('+' in predicate):
            continue

        case_frame2cases = case_frame_dict[predicate]
        case_frames = prune_case_frames(case_frame2cases, cfg.params.case_frame_rate)
        if len(case_frames) == 0:
            L.debug(f'no case frame from {predicate}')
            continue
        ctr.predicate += 1

        # if request and is_intransitive(predicate, cfg.params.sep_rate, cfg.params.case_frame_rate):
        #     intransitive_predicates.add(predicate)

        for case_frame in case_frames:
            pruned_case_frame = {}
            case2arguments = case_frame2cases[case_frame]
            cases = prune_cases(case2arguments, cfg.params.case_rate)

            for case in cases:
                if case not in TRANSLATION_TABLE.keys():
                    L.debug(f'rare case: {case} from {case_frame}')
                    continue
                argument2frequency = case2arguments[case]
                pruned_case_frame[case], case_total = prune_arguments(argument2frequency, cfg.params.argument_rate)
                ctr.argument_frequency += case_total

                for possessive_case in POSSESSIVE_CASES:
                    if case not in possessive_case or f'ノ格~{case}' not in case2arguments.keys():
                        continue

                    possessive_case = f'ノ格~{case}'
                    child, parent = map(lambda x: case2arguments[x]['case_total'], [possessive_case, case])
                    # number of examples of possessive case is large
                    if child / parent > cfg.params.possessive_case_rate:
                        argument2frequency = case2arguments[possessive_case]
                        pruned_case_frame[possessive_case], case_total = prune_arguments(
                            argument2frequency, cfg.params.argument_rate
                        )
                        ctr.argument_frequency += case_total

            if pruned_case_frame:
                pruned[predicate][case_frame] = pruned_case_frame

    if request:
        intransitive_predicates = {
            'お出かけ/おでかけ:動',
            'お疲れさまだ/お疲れさだ:形',
            'デート/でーと:動',
            '上京/じょうきょう:動',
            '世話/せわ:動',
            '休業/きゅうぎょう:動',
            '休館/きゅうかん:動',
            '余りだ/あまりだ:形',
            '僅かだ/わずかだ:形',
            '入院/にゅういん:動',
            '共演/きょうえん:動',
            '再会/さいかい:動',
            '出社/しゅっしゃ:動',
            '受付/うけつけ:動',
            '営業/えいぎょう:動',
            '寝る/ねる:動',
            '御苦労様だ/ごくろうさまだ:形',
            '生意気だ/なまいきだ:形',
            '登れる/のぼれる:動',
            '登校/とうこう:動',
            '眠い/ねむい:形',
            '蒸らす/むらす:動',
            '起き上がる/おきあがる:動',
            '退院/たいいん:動',
            '逝ける/いける?行ける/いける?生ける/いける:動',
            '遠征/えんせい:動',
        }

    return pruned, intransitive_predicates, ctr


def compute_threshold(cfs: list[dict[str, ...]], case_frame_rate: float) -> int:
    cumsum, total = 0, sum(cf['count'] for cf in cfs)
    for idx, cf in enumerate(cfs):
        cumsum += cf['count']
        if (cumsum / total) >= case_frame_rate:  # cut off a long tail
            return idx + 1
    return 0


def compute_case_total(case: str, cf: dict[str, ...]) -> int:
    if case in cf['countTotals'].keys():
        case_total = cf['countTotals'][case]
        # if the most frequent case is "<時間>", ignore the frequency
        if sorted(cf['countmaps'][case].items(), key=lambda x: x[1], reverse=True)[0][0] == '<時間>':
            case_total -= cf['countmaps'][case]['<時間>']
        return case_total
    else:
        return 0


def is_less_frequent_objective_case(cf: dict[str, ...]) -> bool:
    case_totals = [
        compute_case_total(case, cf) for case in ['ニ格', 'デ格', 'ト格', 'カラ格', 'マデ格', 'ヨリ格', 'ヘ格']
    ]
    # sum up "ニ格" and "ヘ格" (treat "ニ格" and "ヘ格" equally) because their deep cases are often similar
    case_totals = [
        case_total + case_totals[-1] if idx == 0 else case_total
        for idx, case_total in enumerate(case_totals[:-1])
    ]
    return all(case_total <= (cf['count'] * 0.8) for case_total in case_totals)


def is_intransitive(predicate: str, sep_rate: float, case_frame_rate: float) -> bool:
    """Return whether a predicate is intransitive or not

    Note:
        - How to determine whether a predicate is intransitive or not
            - We regard the predicates satisfying the following conditions as intransitive
              (and core events consisting of single-word predicates)
                - obj_cond: total frequency of {ニ&ヘ, デ, ト, ヨリ, カラ, マデ}格 is
                            less than or equal to 80% of the total frequency of the case frame
                - sep_cond: percentage of sep3 is greater than or equal to 80%
                - type_cond: predicate is not 判定詞 (a kind of copula)

              "sep" is assigned to each case frame, which represents the following
                  - sep1: examples are transitive
                  - sep2: subject is often not a person (e.g. 叶う (be fulfilled), 混ざる (get mixed))
                  - sep3: subject is often a person (e.g. 寝る (sleep), 嬉しい (be glad))
              We focus on "sep" of top-10 frequent case frames (as infrequent ones are not informative)
    """

    try:
        # pseudo-data
        cfs = [
            {
                'count': 22700,
                'info': [f'origname\tsep1:{predicate}:1'],
                'countmaps': {
                    'ガ格': {
                        'アーティスト/あーてぃすと': 100
                    },
                    'ヲ格': {
                        'イベント/いべんと': 10000,
                        'ライブ/らいぶ': 7500,
                        '説明/せつめい+会/かい': 5000
                    },
                    'ニ格': {
                        'むけ/むけ': 100
                    }
                },
                'countTotals': {
                    'ガ格': 100,
                    'ヲ格': 22500,
                    'ニ格': 100
                }
            }
        ]

        type_cond = not ('判' in cfs[0]['info'][0])  # cf_type = cfs[0]['info'][0]
        top_cfs, count = cfs[:compute_threshold(cfs, case_frame_rate)], 0
        for cf in top_cfs:
            sep_cond = (cf['info'][0].split('\t')[1][:4] == 'sep3')
            obj_cond = is_less_frequent_objective_case(cf)

            if obj_cond and sep_cond and type_cond:
                count += 1
        sleep(0.1)
        return (count / len(top_cfs)) >= sep_rate
    except Exception as e:
        trace_back(e, L.error)
        return False


def process_argument(argument: str) -> Generator[str, None, None]:
    # regarding single-word argument, its modifier is connected with "+" (e.g. 商品+券、事務+所)
    if '+' in argument:
        # e.g. '日本/にほん+人/じん?人/ひと' -> [['日本/にほん'], ['人/じん', '人/ひと']]
        arguments = [word.split('?') for word in argument.split('+')]

        # e.g. [['日本/にほん'], ['人/じん', '人/ひと']] -> [('日本/にほん', '人/じん'), ('日本/にほん', '人/ひと')]
        for produced in product(*arguments):
            yield '+'.join(produced)
    else:
        # if there is ambiguity in the reading or surface expression of argument, the candidates are connected with "?"
        # (e.g. 人/じん?人/ひと -> ['人/じん', '人/ひと'])
        if '?' in argument:
            candidates = argument.split('?')
            for candidate in candidates:
                yield candidate
        else:
            yield argument


# return a string of arguments and case delimited by comma
def create_prefix(case: str, case2processed: dict[str, list[str]]) -> str:
    if case == 'ノ格~判':
        return f'{"|".join(case2processed[case])},{TRANSLATION_TABLE[case]},'
    elif 'ノ格' in case:
        return ''
    else:
        prefix, possessive_case = '', f'ノ格~{case}'
        if possessive_case in case2processed.keys():
            prefix += f'{"|".join(case2processed[possessive_case])},{TRANSLATION_TABLE[possessive_case]},'
        prefix += f'{"|".join(case2processed[case])},{TRANSLATION_TABLE[case]},'
        return prefix


def translate(
    predicate: str,
    ordered_cases: list[str],
    case2processed: dict[str, list[str]],
    removed_case: str = None
) -> str:
    core_event = ''
    copied = [case for case in ordered_cases]
    if removed_case:
        copied.remove(removed_case)

    for case in copied:
        core_event += create_prefix(case, case2processed)
    core_event += predicate
    return core_event  # e.g. {周辺/しゅうへん|地域/ちいき},の/の,{スポット/すぽっと},ヲ,探す/さがす


@decorator(L.info, 'acquisition')
def acquire_core_events(pruned: dict[str, ...], intransitive_predicates: set[str]) -> list[str]:
    core_events = []
    for predicate, case_frame2cases in pruned.items():
        for case_frame, case2arguments in case_frame2cases.items():
            case2processed = ddict(list)
            for case, arguments in case2arguments.items():
                for argument, _ in arguments:
                    for processed in process_argument(argument):
                        case2processed[case].append(processed)

            cases = list(case2processed.keys())
            ordered_cases = [cases[cases.index(case)] for case in TRANSLATION_TABLE.keys() if case in cases]

            # treat "ニ格" and "ヘ格" equally
            if 'ニ格' in cases and 'ヘ格' in cases:
                core_events.append(translate(predicate, ordered_cases, case2processed, removed_case='ニ格'))
                core_events.append(translate(predicate, ordered_cases, case2processed, removed_case='ヘ格'))
            else:
                core_events.append(translate(predicate, ordered_cases, case2processed))
    core_events += [
        intransitive_predicate for intransitive_predicate in sorted(intransitive_predicates)
        if not intransitive_predicate.endswith(':判')
    ]
    return core_events


def count_core_events(core_events: list[str]) -> int:
    bar = tqdm(core_events)
    max_count, total = 0, 0
    for core_event in bar:
        words = core_event.split(',')
        if len(words) == 1:
            total += 1
            continue
        arguments, cases, predicate = words[0:-1:2], words[1:-1:2], words[-1]
        arguments = list(map(lambda x: x.split('|'), arguments))

        count = len(list(product(*arguments)))
        if count > max_count:
            max_count = count
        total += count
        bar.set_postfix({'max_count': max_count, 'total': total})
    return total


def show_statistics(
    case_frame_dict: dict[str, ...],
    pruned: dict[str, ...],
    intransitive_predicates: set[str],
    ctr,
    core_events: list[str]
) -> None:
    num_non_target_predicates = sum('+' in predicate for predicate in case_frame_dict.keys())
    num_predicates = len(case_frame_dict)
    coverage = ctr.argument_frequency / ctr.predicate_frequency
    L.info(f'intransitive predicates: {json.dumps(sorted(list(intransitive_predicates)), indent=2, ensure_ascii=False)}')
    L.info(f'predicates that contain "+": {num_non_target_predicates:,} / {num_predicates:,}')
    L.info(f'acquired predicates: {len(pruned):,} ({ctr.predicate:,}) / {num_predicates - num_non_target_predicates:,}')
    L.info(f'acquired case frames: {sum(len(case_frame2cases) for case_frame2cases in pruned.values()):,}')
    L.info(f'acquired core events: {count_core_events(core_events):,}')
    L.info(f'coverage (コアイベントの用例数 / 全用例数): {coverage:.3f}')


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--request', action='store_true', help='whether to include intransitive predicates or not')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    output_dir = Path(args.OUTPUT)
    set_file_handlers(L, output_path=output_dir.joinpath('acquire_core_events.log'))

    case_frame_dict = load_case_frame_dict(cfg.path.case_frame_dict)
    pruned, intransitive_predicates, ctr = prune(case_frame_dict, cfg, request=args.request)
    core_events = acquire_core_events(pruned, intransitive_predicates)
    show_statistics(case_frame_dict, pruned, intransitive_predicates, ctr, core_events)

    with open(output_dir.joinpath('core_events.txt'), mode='w') as f:
        lines = '\n'.join(core_event for core_event in core_events)
        f.write(lines)


if __name__ == '__main__':
    main()
