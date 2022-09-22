import json
from argparse import ArgumentParser
from collections import defaultdict as ddict
from enum import IntEnum
from itertools import product

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize

from utils import tqdm, ObjectHook, decorator, trace_back


class Column(IntEnum):
    num_morphemes2 = 0
    core_event1 = 1
    core_event2 = 2
    suffix1 = 3
    suffix2 = 4


class CandidateSelector:
    def __init__(self, args):
        with open(args.CONFIG, mode='r') as f:
            cfg = json.load(f, object_hook=ObjectHook)

        self.encoder = ...
        self.content_word_table, self.core_event_table = self.preprocess(cfg.path.table)
        self.content_word_queries, self.core_event_queries = self.preprocess(args.INPUT)
        self.w2v = self.load_w2v(cfg.path.w2v)

        self.lb0, self.ub0 = cfg.params.length
        lb1, ub1 = cfg.params.context_similarity
        lb2, ub2 = cfg.params.choice_similarity
        self.lb1, self.lb2 = map(lambda x: max(x or -1.0, -1.0), [lb1, lb2])
        self.ub1, self.ub2 = map(lambda x: min(x or 1.0, 1.0), [ub1, ub2])

        self.paraphrases, self.translation_table = self.create_translation_tables(cfg.path.blacklist)

    @decorator(print, 'loading')
    def preprocess(self, input_path: str) -> np.array:
        df = pd.read_csv(
            input_path,
            sep='\t',
            usecols=['num_morphemes2', 'content_words1', 'content_words2', 'core_event_pair']
        )
        core_events1, core_events2 = zip(*map(lambda x: x.split('|'), df['core_event_pair']))

        if self.encoder is ...:
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
            df[['content_words1', 'content_words2']].values, \
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

    def update(self, core_event: str) -> int:
        if core_event not in self.encoder.keys():
            self.encoder[core_event] = len(self.encoder.keys())
        return self.encoder[core_event]

    def create_translation_tables(self, input_path: str) -> tuple[ddict, ddict]:
        paraphrases, translation_table = ddict(lambda: -1), ddict(set)
        with open(input_path, mode='r') as f:
            for line in f:
                switch, core_event1, core_event2 = line.strip().split('|')
                if switch == 'paraphrase':
                    core_event1, core_event2 = map(lambda x: self.update(x), [core_event1, core_event2])
                    paraphrases.update({core_event1: core_event2, core_event2: core_event1})
                elif switch == 'script':
                    for suffix1, suffix2 in product(core_event1.split('-'), core_event2.split('-')):
                        suffix1, suffix2 = map(lambda x: self.update(x), [suffix1, suffix2])
                        translation_table[suffix1].add(suffix2)
                        translation_table[suffix2].add(suffix1)
                else:
                    raise ValueError('invalid value')

        for core_event1, core_event2 in self.core_event_table[:, [1, 2]]:
            translation_table[core_event1].add(core_event2)
            if core_event1 in paraphrases.keys():
                translation_table[paraphrases[core_event1]].add(core_event2)
            if core_event2 in paraphrases.keys():
                translation_table[core_event1].add(paraphrases[core_event2])

        return paraphrases, translation_table

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

    def get_normalized_vectors(self, values: np.array) -> np.array:
        buf = []
        for value in values:
            buf.append(self.convert_content_words_into_vector(eval(value)))
        return normalize(np.stack(buf, axis=0), norm='l2', axis=1)

    def apply_similarity_condition(self) -> np.array:
        buf, chunk_size = [], 10000
        queries1, queries2 = map(
            lambda x: self.get_normalized_vectors(self.content_word_queries[:, x]), [0, 1]
        )
        for start in tqdm(range(0, len(self.content_word_table), chunk_size)):
            rows1, rows2 = map(
                lambda x: self.get_normalized_vectors(
                    self.content_word_table[start: start + chunk_size, x]
                ).transpose(1, 0),
                [0, 1]
            )
            cos_sim1 = np.matmul(queries1, rows1)
            mask = (self.lb1 <= cos_sim1) & (cos_sim1 <= self.ub1)
            cos_sim2 = np.matmul(queries2, rows2)
            mask &= (self.lb2 <= cos_sim2) & (cos_sim2 <= self.ub2)
            buf.append(mask)
        return np.concatenate(buf, axis=1)

    def apply_length_condition(self, cache: np.array) -> np.array:
        for idx, (num_morphemes2, core_event1, core_event2, suffix1, _) in enumerate(tqdm(self.core_event_queries)):
            mask = (
                self.lb0 <= self.core_event_table[:, Column.num_morphemes2] / num_morphemes2
            ) & (
                self.core_event_table[:, Column.num_morphemes2] / num_morphemes2 <= self.ub0
            )
            if self.ub1 < 1.0:
                mask &= ~np.isin(
                    self.core_event_table[:, Column.core_event1],
                    np.array([core_event1, self.paraphrases[core_event1]], dtype='int32')
                )
            if self.ub2 < 1.0:
                mask &= ~np.isin(
                    self.core_event_table[:, Column.core_event2],
                    np.array(
                        [
                            core_event2,
                            *self.translation_table[core_event1],
                            *self.translation_table[self.paraphrases[core_event1]]
                        ],
                        dtype='int32'
                    )
                )
                mask &= ~np.isin(
                    self.core_event_table[:, Column.suffix2],
                    np.array([*self.translation_table[suffix1]], dtype='int32')
                )
                # if '(intransitive)' in encoder[core_event1]:
                #     decoder = {id_: core_event for core_event, id_ in self.encoder.items()}
                #     ids = [id_ for id_ in translation_table[suffix1] if '(intransitive)' not in decoder[id_]]
                #     ref = [decoder[id_] for id_ in translation_table[suffix1] if '(intransitive)' not in decoder[id_]]
                #     indices, *_ = np.where(
                #         np.isin(self.core_event_table[:, Column.suffix2], np.array(ids, dtype='int32'))
                #     )
                #     assert len(indices) == 0
            cache[idx, :] &= mask
        return cache

    def select_candidates(self):
        return self.apply_length_condition(self.apply_similarity_condition())


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    candidate_selector = CandidateSelector(args)
    candidates = candidate_selector.select_candidates()
    np.save(args.OUTPUT, candidates)


if __name__ == '__main__':
    main()
