import re
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict as ddict

import pandas as pd
from ordered_set import OrderedSet
from pyknp_eventgraph import EventGraph

from utils import CustomDict, decorator, get_logger, set_file_handlers, load_parsing_results, trace_back


L = get_logger(__name__)


class Extractor:
    def __init__(self, args):
        self.args = args

        self.normalizer = CustomDict({
            '下さる': '下さい', 'くださる': 'ください', '致す': '致します', 'いたす': 'いたします',
            ('ませ', 'ん', 'だ'): ['ませ', 'ん', 'でした']
        })
        self.special_symbols = '。．、，）〕］｝〉＞》」』】”・：；？！'
        self.columns = [
            'sid',
            'orig',
            'morphemes1',
            'morphemes1_wo_modifier',
            'normalized_morphemes1',
            'normalized_morphemes2',
            'event1',
            'normalized_event1',
            'normalized_event2',
            'num_base_phrases2',
            'num_morphemes2',
            'pas1',
            'pas2',
            'content_words1',
            'ya_content_words1',
            'content_words2',
            'ya_content_words2',
            'type1',
            'type2',
            'rel',
            'rel_surf',
            'reliable',
            'core_event_pair',
            'basic',
            'pou1',
            'pou2',
            'negation1',
            'negation2',
            'potential1',
            'potential2',
            'normalized_predicate1',
            'normalized_predicate2'
        ]

        self.predicate2case_frame_id, self.predicate2cases = self.create_translation_tables(args.core_events)

    @staticmethod
    @decorator(print, 'loading')
    def create_translation_tables(input_path: str) -> tuple[ddict, ddict]:
        """Create translation tables

        Returns:
            predicate2case_frame_id: dict[str, dict[str, dict[str, OrderedSet]]]
                depth1: predicate
                depth2: case
                depth3: argument
                depth4: case frame id
                e.g., {'壊す/こわす': {'ヲ': {'お腹/おなか': {1}, '雰囲気/ふんいき': {2}, ... }}, ... }

            predicate2cases: dict[str, dict[str, set[str]]]
                depth1: predicate
                depth2: case frame id
                depth3: case
                e.g., {'壊す/こわす': {1: {ヲ}, 2: {の/の, ヲ}, ... }, ... }
        """

        predicate2arguments_and_case_pairs = ddict(OrderedSet)
        with open(input_path, mode='r') as f:
            for line in f:
                try:
                    # e.g. ('お腹|体,ヲ', '壊す/こわす')
                    arguments_and_case_pairs, predicate = line.strip().rsplit(',', 1)
                except ValueError:
                    arguments_and_case_pairs, predicate = 'intransitive', line.strip()

                for predicate in predicate.split(':')[0].split('?'):
                    predicate2arguments_and_case_pairs[predicate].add(arguments_and_case_pairs)

        predicate2case_frame_id = ddict(lambda: ddict(lambda: ddict(OrderedSet)))
        predicate2cases = ddict(lambda: ddict(lambda: ddict(set)))
        for predicate, ordered_set in predicate2arguments_and_case_pairs.items():
            for idx, arguments_and_case_pairs in enumerate(ordered_set):
                case_frame_id = idx + 1
                if arguments_and_case_pairs == 'intransitive':  # 自動詞
                    predicate2case_frame_id[predicate]['intransitive'] = ...
                    continue

                words = arguments_and_case_pairs.split(',')
                for arguments, case in zip(words[0::2], words[1::2]):
                    for argument in arguments.split('|'):
                        predicate2case_frame_id[predicate][case][argument].add(case_frame_id)
                predicate2cases[predicate][case_frame_id] = words[1::2]

        return predicate2case_frame_id, predicate2cases

    def modify_predicate(self, event, include_modifiers: bool = True) -> tuple[str, bool]:
        normalized_morphemes = event.normalized_mrphs_(include_modifiers=include_modifiers).split(' ')
        normalized_morphemes[-1] = self.normalizer[normalized_morphemes[-1]]
        normalized_morphemes[-3:] = self.normalizer[tuple(normalized_morphemes[-3:])]
        if normalized_morphemes[-1] == 'ない':
            surf = event.surf.rstrip(self.special_symbols)
            if mo := re.fullmatch(r'^.+ない(?P<modality>[かと])$', surf):
                normalized_morphemes.append(mo.group('modality'))
            elif surf.endswith('なきゃ'):
                normalized_morphemes[-1] = 'なきゃ'

        adjective = False
        try:
            morphemes = [
                morpheme
                for base_phrase in event.get_constituent_base_phrases()
                for morpheme in base_phrase.tag.mrph_list()
            ]
            morpheme1, morpheme2 = [
                (m1, m2) for m1, m2 in zip(morphemes[:-1], morphemes[1:])
                if m2.genkei == normalized_morphemes[-1]
            ][-1]
            if morpheme2.bunrui == 'サ変名詞':
                # 一定する -> 一定だ
                if morpheme2.genkei in {'一定', '予定', '全開', '原因', '安心', '山積み', '満載', '満開', '評判'}:
                    normalized_morphemes.append('だ')
                    adjective = True
                else:
                    normalized_morphemes.append('をする' if morpheme1.genkei == 'の' else 'する')
        except IndexError:
            pass

        return ' '.join(normalized_morphemes), adjective

    # get head morpheme in base phrase
    @staticmethod
    def get_head_morpheme(morphemes: list):
        head_morpheme = None
        for morpheme in morphemes:
            if '内容語' in morpheme.fstring and head_morpheme is None:
                head_morpheme = morpheme
            if '準内容語' in morpheme.fstring:
                head_morpheme = morpheme

        if head_morpheme:
            return head_morpheme

        return morphemes[0]

    def find_noisy_base_phrases(self, event2, event1) -> tuple[str, int]:
        """find noisy base phrases

        NOTE:
            - Due to dependency parsing results and the specifications of EventGraph,
              noisy base phrases are sometimes appended to latter event (event2)
              e.g. お腹が空いたので -> というか、|ご飯を食べる
            - The modification makes event pairs a bit ungrammatical,
              so remove such noisy base phrases based on rules
        """

        base_phrases = event2.get_constituent_base_phrases()
        # find sequence of base phrases connected with "|"
        for idx, (base_phrase1, base_phrase2) in enumerate(zip(base_phrases[:-1], base_phrases[1:])):
            if base_phrase2.tid - base_phrase1.tid > 1:
                appendix = [
                    self.get_head_morpheme(base_phrase.tag.mrph_list()) for base_phrase in base_phrases[:idx + 1]
                ]
                break
        else:
            return '', 0

        if all(
            head_morpheme.hinsi in {'感動詞', '指示詞', '助詞', '接続詞', '判定詞', '副詞', '特殊'} or
            head_morpheme.bunrui in {'形式名詞', '数詞'} or
            head_morpheme.repname in {'言う/いう', '因む/ちなむ'} for head_morpheme in appendix
        ):
            old = ''.join(base_phrase.surf for base_phrase in base_phrases[:len(appendix)])
            new = '\033[38;5;219m' + old + '\033[0m'
            L.info(f'{event1.surf} -> {event2.surf.replace(old, new, 1)}')
            return ' '.join(
                morpheme.midasi
                for base_phrase in base_phrases[:len(appendix)]
                for morpheme in base_phrase.tag.mrph_list()
            ), len(appendix)
        elif (
            appendix[0].hinsi in {'感動詞', '助詞', '接続詞', '判定詞', '特殊'} or
            base_phrases[0].surf.rstrip(self.special_symbols) == 'てか'
        ):
            old = base_phrases[0].surf
            new = '\033[38;5;219m' + old + '\033[0m'
            L.info(f'{event1.surf} -> {event2.surf.replace(old, new, 1)}')
            return ' '.join(morpheme.midasi for morpheme in base_phrases[0].tag.mrph_list()), 1
        else:
            return '', 0

    @staticmethod
    def get_pas_string(pas) -> str:
        cases = [
            case for case in [
                'ガ', 'ヲ', 'ニ', 'デ', 'ト', 'ヘ', 'カラ', 'マデ', 'ヨリ', '修飾', '時間', 'ニクラベル', 'ニトル', 'ニヨル'
            ] if case in pas.arguments.keys() and pas.arguments[case][0].head_reps
        ]
        ret = []
        for case in cases:
            children = pas.arguments[case][0].children
            if children and children[0]['possessive'] and children[0]["head_reps"]:
                ret.extend([f'{children[0]["head_reps"].replace(" ", "+")}', 'の/の'])
            ret.extend([f'{pas.arguments[case][0].head_reps.replace(" ", "+")}', case])
        ret.append(pas.predicate.reps.replace(' ', '+'))
        return ' '.join(ret)

    @staticmethod
    def get_content_words(base_phrases: list, standard: str = None) -> list[str]:
        content_words = []
        for base_phrase in base_phrases:
            for morpheme in base_phrase.tag.mrph_list():
                if standard == 'own':
                    cond = morpheme.repname == 'ない/ない'
                    cond |= ('<内容語>' in morpheme.fstring and morpheme.hinsi in {'名詞', '動詞', '形容詞'})
                elif standard == 'EventGraph':
                    cond = any(tag in morpheme.fstring for tag in ['<内容語>', '<準内容語>'])
                else:
                    raise ValueError('invalid standard')

                if cond:
                    content_words.append(morpheme.repname or f"{morpheme.midasi}/{morpheme.midasi}")

        return content_words

    @staticmethod
    def is_pronoun_or_undefined(base_phrases: list) -> bool:
        return any(
            morpheme.hinsi in {'指示詞', '未定義語'}
            for base_phrase in base_phrases
            for morpheme in base_phrase.tag.mrph_list()
        )

    @staticmethod
    def is_potential(pas) -> bool:
        cond = ('<可能表現>' in pas.predicate.head.fstring)
        if pas.pas is None:
            cond &= True
        else:
            predicate = pas.pas.tag_list[pas.pas.tid]
            fstrings = [morpheme.fstring for morpheme in predicate.mrph_list()]
            # remove such as 〜かもしれない and 〜していただける
            cond &= not any(
                '<付属>' in fstring and re.search(r'<可能動詞:(?P<basic>.+?)>', fstring) is not None
                for fstring in fstrings
            )
        return cond

    def get_normalized_predicate(self, predicate) -> str:
        head_morpheme = self.get_head_morpheme(predicate.head.mrph_list())
        mo = re.search(r'<可能動詞:(?P<basic>.+?)>', head_morpheme.fstring)
        if mo:
            # head morpheme is in the potential  e.g. 会える, 折れる (「折ることができる」 is more common)
            return mo.group('basic')
        else:
            # predicate contains function words representing potential  e.g. 運転+できる, 食べる+ことが+できる
            return head_morpheme.repname

    def find_core_event(self, words: list[str]) -> str:
        """Return core event in PAS

        Examples:
            pas = ['場/ば', 'の/の', '雰囲気/ふんいき', 'ヲ', '壊す/こわす']
            predicate = '壊す/こわす'
            cases = ['の/の', 'ヲ']
            arguments = ['場/ば', '雰囲気/ふんいき']

            predicate2case_frame_id['壊す/こわす'] = {
                'ヲ': {
                    'お腹/おなか': {1},
                    '雰囲気/ふんいき': {2},
                    ...
                },
                ...
            }
            predicate2cases['壊す/こわす'] = {
                1: {'ヲ'},
                2: {'の/の', 'ヲ'},
                ...
            }

            predicate2case_frame_id['行う/おこなう']['の/の']['場/ば'] = {2}
            predicate2case_frame_id['行う/おこなう']['ヲ']['雰囲気/ふんいき'] = {2}
            matched = {2: ['の/の', 'ヲ']}

            From the result of matched, it is determined to contain core event
            '場/ば,の/の,雰囲気/ふんいき,ヲ,壊す/こわす (case_frame_id = 2)'
            -> return '場/ば,の/の,雰囲気/ふんいき,ヲ,壊す/こわす'
        """

        predicate, cases, arguments = words[-1], words[1:-1:2], words[0:-1:2]

        case2argument, matched = {}, ddict(list)
        if predicate in self.predicate2case_frame_id.keys():
            for argument, case in zip(arguments, cases):
                case2argument[case] = argument
                if (
                    case in self.predicate2case_frame_id[predicate].keys() and
                    argument in self.predicate2case_frame_id[predicate][case].keys()
                ):
                    for case_frame_id in self.predicate2case_frame_id[predicate][case][argument]:
                        # 基本イベント集合を検索し、一致する格と項があったので、そのIDと格を記録する
                        matched[case_frame_id].append(case)

        if len(matched) > 0:
            for case_frame_id in matched.keys():
                # PAS ⊃ 基本イベント を表す条件 (基本イベント ⊃ PAS ならば,0より大きくなる)
                if len(set(self.predicate2cases[predicate][case_frame_id]) - set(matched[case_frame_id])) == 0:
                    argument_and_case_pairs = [f'{case2argument[case]},{case}' for case in matched[case_frame_id]]
                    return f'{",".join(argument_and_case_pairs)},{predicate}'

        # check at the end whether a single-word core event is contained
        if 'intransitive' in self.predicate2case_frame_id[predicate].keys():
            return f'(intransitive){predicate}'
        return ''

    def complement(
        self,
        predicate2: str,
        cases1: list[str],
        arguments1: list[str]
    ) -> str:
        """Investigate whether some arguments in former event (event1) can be complemented to latter event (event2)
           (if latter event is single-word core event)

        Examples:
            core_event_pair = 電池/でんち ガ 切れる/きれる -> 取り替える/とりかえる
            predicate2 = '取り替える/とりかえる'
            cases1 = ['ガ']
            arguments1 = ['電池/でんち']

            predicate2case_frame_id['取り替える/とりかえる'] = {
                'ヲ': {
                    ...
                    '電池/でんち': {2},
                    ...
                },
                ...
            }

            Since we can find '電池/でんち' in the examples of ヲ格 of the second case frame of '取り替える',
            '電池/でんち,ヲ' is complemented to predicate2
        """

        for argument1, case1 in zip(arguments1, cases1):
            for case2, argument2case_frame_id in self.predicate2case_frame_id[predicate2].items():
                # if case1 != 'の/の' and case2 != 'の/の': <- ノ格の項を完全に無視する場合
                if case2 != 'の/の' and argument1 in argument2case_frame_id.keys():  # ノ格は補完しない
                    return f'(complemented){argument1},{case2},{predicate2}'
        return ''

    def find_core_event_pair(
        self,
        pas_strings: tuple[str, str],
        negation1: bool,
        negation2: bool
    ) -> str:
        words1, words2 = map(lambda x: x.split(' '), pas_strings)
        assert all(
            case in {
                'ガ', 'デ', 'ト', 'ニ', 'ヘ', 'ヲ', 'カラ', 'マデ', 'ヨリ', 'の/の',
                '修飾', '時間', 'ニクラベル', 'ニトル', 'ニヨル'
            } for case in words1[1:-1:2] + words2[1:-1:2]
        ), 'invalid pas'

        core_event1, core_event2 = map(lambda x: self.find_core_event(x), [words1, words2])

        predicate1, predicate2, cases1, arguments1 = (
             words1[-1],
             words2[-1],
             words1[1:-1:2],
             words1[0:-1:2]
        )
        # e.g. 電球,ガ,切れる -> 取り替える
        if core_event1 != '' and core_event2 == '' and len(words2) == 1 and predicate1 != predicate2:
            core_event2 = self.complement(predicate2, cases1, arguments1)
        core_event1 += 'n' * int(negation1 and core_event1 != '')
        core_event2 += 'n' * int(negation2 and core_event2 != '')
        return f'{core_event1}|{core_event2}'

    def extract_event_pair(self, evg: EventGraph) -> list[OrderedDict]:
        buf = []
        for rel in evg.relations:
            event1, event2 = map(lambda x: getattr(rel, x), ['modifier', 'head'])
            include_modifiers = (rel.label not in {'補文', '連体修飾'})
            morphemes1 = event1.mrphs_(include_modifiers=True)
            normalized_morphemes1, adjective1 = self.modify_predicate(event1, include_modifiers=True)
            normalized_morphemes2, adjective2 = self.modify_predicate(event2, include_modifiers=include_modifiers)

            base_phrases1, base_phrases2 = map(lambda x: x.get_constituent_base_phrases(), [event1, event2])
            num_base_phrases2 = len(base_phrases2)

            if include_modifiers:
                noisy_prefix, subtrahend = self.find_noisy_base_phrases(event2, event1)
                normalized_morphemes2 = normalized_morphemes2.replace(noisy_prefix, '', 1).lstrip()
                num_base_phrases2 -= subtrahend
            else:
                for idx, (base_phrase1, base_phrase2) in enumerate(zip(base_phrases2[:-1], base_phrases2[1:])):
                    if base_phrase2.tid - base_phrase1.tid > 1:
                        noisy_prefix = ' '.join(
                            morpheme.midasi
                            for base_phrase in base_phrase2[:idx + 1]
                            for morpheme in base_phrase.tag.mrph_list()
                        )
                        normalized_morphemes2 = normalized_morphemes2.replace(noisy_prefix, '', 1).lstrip()
                        num_base_phrases2 -= (idx + 1)
                        break

            if event1.mrphs_with_mark == '' or event2.mrphs_with_mark == '' or num_base_phrases2 <= 0:
                continue

            pas_string1, pas_string2 = map(lambda x: self.get_pas_string(x.pas), [event1, event2])
            core_event_pair = self.find_core_event_pair(
                (pas_string1, pas_string2), event1.features.negation, event2.features.negation
            )
            potential1, potential2 = map(lambda x: self.is_potential(x.pas), [event1, event2])

            event_pair = OrderedDict([
                ('sid', event1.sentence.sid),
                ('orig', event1.sentence.surf),
                ('morphemes1', morphemes1),
                ('morphemes1_wo_modifier', event1.mrphs),
                ('normalized_morphemes1', normalized_morphemes1),
                ('normalized_morphemes2', normalized_morphemes2),
                ('event1', morphemes1.replace(' ', '')),
                ('normalized_event1', normalized_morphemes1.replace(' ', '')),
                ('normalized_event2', normalized_morphemes2.replace(' ', '')),
                ('num_base_phrases2', num_base_phrases2),
                ('num_morphemes2', len(normalized_morphemes2.split(' '))),
                ('pas1', pas_string1),
                ('pas2', pas_string2),
                ('content_words1', self.get_content_words(base_phrases1, standard='own')),
                ('ya_content_words1', self.get_content_words(base_phrases1, standard='EventGraph')),
                ('content_words2', self.get_content_words(base_phrases2, standard='own')),
                ('ya_content_words2', self.get_content_words(base_phrases2, standard='EventGraph')),
                ('type1', '形' if adjective1 else event1.pas.predicate.type_),
                ('type2', '形' if adjective2 else event2.pas.predicate.type_),
                ('rel', rel.label),
                ('rel_surf', rel.surf),
                ('reliable', rel.reliable),
                ('core_event_pair', core_event_pair),
                ('basic', re.search(r'.+\|.+', core_event_pair) is not None),
                ('pou1', self.is_pronoun_or_undefined(base_phrases1)),
                ('pou2', self.is_pronoun_or_undefined(base_phrases2)),
                ('negation1', event1.features.negation),
                ('negation2', event2.features.negation),
                ('potential1', potential1),
                ('potential2', potential2),
                ('normalized_predicate1', self.get_normalized_predicate(event1.pas.predicate) if potential1 else ''),
                ('normalized_predicate2', self.get_normalized_predicate(event2.pas.predicate) if potential2 else '')
            ])
            assert all(key == column for key, column in zip(event_pair.keys(), self.columns))
            buf.append(event_pair)
        return buf

    def extract_event_pairs(self) -> pd.DataFrame:
        buf = []
        for blist in load_parsing_results(self.args.INPUT, buf_size=1, silent=self.args.silent):
            try:
                evg = EventGraph.build(blist)
                buf.extend(self.extract_event_pair(evg))
            except Exception as e:
                trace_back(e, print)
                continue
        return pd.DataFrame(buf, columns=self.columns)


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('--core-events', type=str, help='path to list of core events')
    parser.add_argument('--silent', action='store_true', help='whether to print progress bar')
    args = parser.parse_args()

    set_file_handlers(L)

    extractor = Extractor(args)
    df = extractor.extract_event_pairs()
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()
