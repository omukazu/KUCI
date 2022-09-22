import json
from argparse import ArgumentParser
from enum import IntEnum

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from pyknp import KNP
from sklearn.preprocessing import normalize

from utils import tqdm, ObjectHook, decorator, trace_back


class Column(IntEnum):
    num_morphemes2 = 0
    core_event1 = 1
    core_event2 = 2
    suffix1 = 3
    suffix2 = 4


class InteractiveCandidateSelector:
    def __init__(self, args):
        with open(args.CONFIG, mode='r') as f:
            cfg = json.load(f, object_hook=ObjectHook)

        self.encoder = ...
        self.content_word_table, self.core_event_table = self.preprocess(cfg.path.table)
        self.w2v = self.load_w2v(cfg.path.w2v)

        self.lb0, self.ub0 = cfg.params.length
        lb1, ub1 = cfg.params.context_similarity
        lb2, ub2 = cfg.params.choice_similarity
        self.lb1, self.lb2 = map(lambda x: max(x or -1.0, -1.0), [lb1, lb2])
        self.ub1, self.ub2 = map(lambda x: min(x or 1.0, 1.0), [ub1, ub2])

    @decorator(print, 'loading')
    def preprocess(self, input_path: str) -> np.array:
        df = pd.read_csv(
            input_path,
            sep='\t',
            usecols=[
                'event1',
                'normalized_event2',
                'num_morphemes2',
                'content_words1',
                'content_words2',
                'core_event_pair'
            ]
        )
        core_events1, core_events2 = zip(*map(lambda x: x.split('|'), df['core_event_pair']))

        if self.encoder is Ellipsis:
            core_events = set(core_events1) | set(core_events2)
            core_events |= {','.join(core_evt.split(',')[-2:]) for core_evt in core_events}
            self.encoder = {core_event: idx for idx, core_event in enumerate(core_events)}

        df = df.assign(
            core_event1=[self.encoder[core_event] for core_event in core_events1],
            core_event2=[self.encoder[core_event] for core_event in core_events2],
            suffix1=[self.encoder[','.join(core_event.split(',')[-2:])] for core_event in core_events1],
            suffix2=[self.encoder[','.join(core_event.split(',')[-2:])] for core_event in core_events2],
        ).drop(['core_event_pair'], axis=1)

        return \
            df[['content_words1', 'content_words2', 'event1', 'normalized_event2']].values, \
            df[[column.name for column in Column]].values.astype('int32')

    @staticmethod
    @decorator(print, 'loading')
    def load_w2v(input_path: str) -> dict[str, np.array]:
        w2v = KeyedVectors.load_word2vec_format(input_path, binary=True, unicode_errors='ignore')
        dis_amb, amb = {}, {}
        for key in w2v.vocab.keys():
            if '?' in key:
                amb.update({rep: w2v.get_vector(key) for rep in key.split('?')})
            else:
                dis_amb[key] = w2v.get_vector(key)
        amb.update(dis_amb)  # overwrite unambiguous vector
        return amb

    @staticmethod
    def get_content_words(morphemes: list) -> list[str]:
        content_words = []
        for morpheme in morphemes:
            cond = morpheme.repname == 'ない/ない'
            cond |= ('<内容語>' in morpheme.fstring and morpheme.hinsi in {'名詞', '動詞', '形容詞'})
            if cond:
                content_words.append(morpheme.repname or f"{morpheme.midasi}/{morpheme.midasi}")
        return content_words

    def convert_content_words_into_vector(self, content_words: list[str]) -> np.array:
        if len(content_words) > 0:
            content_word_vectors = [
                self.w2v[content_word]
                if content_word in self.w2v.keys() else
                self.w2v['<UNK>']
                for content_word in content_words
            ]
            return sum(content_word_vectors) / len(content_word_vectors)
        else:
            return np.zeros(self.w2v['<UNK>'].shape)

    @decorator(print, 'converting')
    def get_normalized_vectors(self, values: np.array) -> np.array:
        buf = []
        for value in values:
            buf.append(self.convert_content_words_into_vector(eval(value)))
        return normalize(np.stack(buf, axis=0), norm='l2', axis=1)

    def interactive(self):
        knp = KNP(option='-tab -assignf')

        table1, table2 = map(
            lambda x: self.get_normalized_vectors(
                self.content_word_table[:, x]
            ).transpose(1, 0),
            [0, 1]
        )

        while True:
            event_pair = input('input event pair (delimited by "|")')
            try:
                event1, event2 = event_pair.split('|')
                morphemes1, morphemes2 = map(lambda x: knp.parse(x).mrph_list(), [event1, event2])
                content_words1, content_words2 = map(lambda x: self.get_content_words(x), [morphemes1, morphemes2])
                query1, query2 = map(
                    lambda x: normalize(
                        self.convert_content_words_into_vector(x).reshape(1, -1),
                        norm='l2',
                        axis=1
                    ),
                    [content_words1, content_words2]
                )
                cos_sim1 = np.matmul(query1, table1)
                mask = (self.lb1 <= cos_sim1) & (cos_sim1 <= self.ub1)
                cos_sim2 = np.matmul(query2, table2)
                mask &= (self.lb2 <= cos_sim2) & (cos_sim2 <= self.ub2)
                num_morphemes2 = len(morphemes2)
                mask &= (
                    self.lb0 <= self.core_event_table[:, Column.num_morphemes2] / num_morphemes2
                ) & (
                    self.core_event_table[:, Column.num_morphemes2] / num_morphemes2 <= self.ub0
                )
                for idx, flag in enumerate(mask.squeeze(0)):
                    if flag:
                        v1, v2, v3, v4 = self.content_word_table[idx]
                        print(f'・{v3}|{v4}\n({v1}|{v2})')
            except Exception as e:
                trace_back(e, print)
                continue


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    args = parser.parse_args()

    interactive_candidate_selector = InteractiveCandidateSelector(args)
    interactive_candidate_selector.interactive()


if __name__ == '__main__':
    main()
