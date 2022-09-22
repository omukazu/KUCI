from argparse import ArgumentParser

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from tqdm import tqdm


tqdm.pandas()
FASTEN = [
    '気/き,ニ',
    '部品/ぶひん,ヲ',
    '部分/ぶぶん,ヲ'
]
GET_USED_TO = [
    '仕事/しごと,ニ',
    '味/あじ,ニ',
    '寒い/さむい+さ/さ,ニ',
    '携帯/けいたい,ニ',
    '操作/そうさ,ニ',
    '文章/ぶんしょう,ニ',
    '生活/せいかつ,ニ'
]
LOSE = [
    '父/ちち,ヲ',
    '父親/ちちおや,ヲ',
    '母/はは,ヲ',
    '母親/ははおや,ヲ',
    '人/ひと,ヲ',
    '両親/りょうしん,ヲ'
]
STING = [
    '心/こころ,ヲ',
    '肌/はだ,ヲ',
    '頰/ほお,ヲ'
]


def get_argument2argument(argument_and_case_pairs: OrderedSet) -> dict[str, str]:
    if '傍/そば,ヲ' in argument_and_case_pairs:
        return {'傍/そば': '蕎麦/そば'}
    return {}


def get_predicate2predicate(predicate: str, argument_and_case_pairs: OrderedSet) -> dict[str, str]:
    if predicate == '係る/かかる':
        return {'係る/かかる': '掛かる/かかる'}
    elif predicate == '刳る/くる':
        return {'刳る/くる': '来る/くる'}
    elif predicate == '亡くす/なくす' and all(pair not in argument_and_case_pairs for pair in LOSE):
        return {'亡くす/なくす': '無くす/なくす'}
    elif predicate == '亡くなる/なくなる':
        return {'亡くなる/なくなる': '無くなる/なくなる'}
    elif predicate == '降りる/おりる' and '階段/かいだん,ヲ' in argument_and_case_pairs:
        return {'降りる/おりる': '下りる/おりる'}
    elif predicate == '登る/のぼる' and '階段/かいだん,ヲ' in argument_and_case_pairs:
        return {'登る/のぼる': '上る/のぼる'}
    elif predicate == '当てる/あてる' and '支払い/しはらいv,ニ' in argument_and_case_pairs:
        return {'当てる/あてる': '充てる/あてる'}
    elif predicate in {'好く/すく', '空く/あく'} and 'お腹/おなか,ガ' in argument_and_case_pairs:
        return {'好く/すく': '空く/すく'} if predicate == '好く/すく' else {'空く/あく': '空く/すく'}
    elif predicate == '刺す/さす' and all(pair not in argument_and_case_pairs for pair in STING):
        return {'刺す/さす': '差す/さす'}
    elif predicate == '慣れる/なれる' and all(pair not in argument_and_case_pairs for pair in GET_USED_TO):
        return {'慣れる/なれる': 'なれる/なれる'}
    elif predicate == '取る/とる' and '写真/しゃしん,ヲ' in argument_and_case_pairs:
        return {'取る/とる': '撮る/とる'}
    elif predicate == '弾ける/ひける' and '気/き,ガ' in argument_and_case_pairs:
        return {'弾ける/ひける': '引ける/ひける'}
    elif predicate == '富める/とめる':
        return (
            {'富める/とめる': '留める/とめる'}
            if any(pair in argument_and_case_pairs for pair in FASTEN) else
            {'富める/とめる': '止める/とめる'}
        )
    return {}


def update_content_words(content_words: list[str], translation_tables: np.array) -> list[str]:
    predicate2predicate, argument2argument = translation_tables
    if predicate2predicate:
        for idx in range(len(content_words) - 1, -1, -1):
            if content_words[idx] in predicate2predicate.keys():
                content_words[idx] = predicate2predicate[content_words[idx]]
                break

    if argument2argument:
        for idx in range(len(content_words) - 1, -1, -1):
            if content_words[idx] in argument2argument.keys():
                content_words[idx] = argument2argument[content_words[idx]]
                break

    return content_words


def update_core_event(
    core_event: list[str],
    predicate2predicate: dict[str, str],
    argument2argument: dict[str, str],
) -> list[str]:
    if core_event[-1] in predicate2predicate.keys():
        core_event[-1] = predicate2predicate[core_event[-1]]
    for idx, argument in enumerate(core_event[0:-1:2]):
        if argument in argument2argument.keys():
            core_event[idx * 2] = argument2argument[argument]
    core_event[-1] = core_event[-1].replace('(single)', '(intransitive)')
    return core_event


def update_core_event_pair(core_event_pair: str, translation_tables: np.array) -> str:
    core_event1, core_event2 = map(lambda x: x.split(','), core_event_pair.split('|'))
    buf = []
    for core_event, indices in [(core_event1, [0, 2]), (core_event2, [1, 3])]:
        buf.append(update_core_event(core_event, *translation_tables[indices]))
    return '|'.join(','.join(core_event) for core_event in buf)


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input/output')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')

    buf = []
    for core_event_pair in df['core_event_pair']:
        core_event1, core_event2 = map(lambda x: x.split(','), core_event_pair.split('|'))
        argument_and_case_pairs1, argument_and_case_pairs2 = map(
            lambda x: OrderedSet([f'{argument},{case}' for argument, case in zip(x[0:-1:2], x[1:-1:2])]),
            [core_event1, core_event2]
        )
        buf.append([
            get_predicate2predicate(core_event1[-1], argument_and_case_pairs1),
            get_predicate2predicate(core_event2[-1], argument_and_case_pairs2),
            get_argument2argument(argument_and_case_pairs1),
            get_argument2argument(argument_and_case_pairs2)
        ])
    idx2translation_tables = np.array(buf)

    df['idx'] = df.index.values
    for column in ['content_words1', 'content_words2', 'ya_contend_words1', 'ya_content_words2']:
        df[column] = df[column].map(eval)
        idx2column = df[column].to_dict()
        indices = [0, 2] if column.endswith('1') else [1, 3]
        df[column] = df.idx.map(lambda x: update_content_words(idx2column[x], idx2translation_tables[x, indices]))
    else:
        column = 'core_event_pair'
        idx2core_event_pair = df[column].to_dict()
        df[column] = df.idx.map(lambda x: update_core_event_pair(idx2core_event_pair[x], idx2translation_tables[x]))
    df.drop(['idx'], axis=1, inplace=True)

    df.to_csv(args.INPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
