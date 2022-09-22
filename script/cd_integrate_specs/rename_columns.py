from argparse import ArgumentParser

import pandas as pd


COLUMNS = [
    'sid',
    'orig',
    'morphemes1',
    'morphemes1_wo_modifier',
    'normalized_morphemes1',
    'normalized_morphemes2',
    'event1',
    'old_normalized_event1',
    'normalized_event1',
    'old_normalized_event2',
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


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    parser.add_argument('-v', type=int, help='number of version')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')
    if args.v in {1, 2}:
        df.rename(
            columns={
                'sentence': 'orig',
                'separated_former': 'morphemes1',
                'separated_query': 'normalized_morphemes1',
                'separated_latter': 'normalized_morphemes2',
                'former': 'event1',
                'old_query': 'old_normalized_event1',
                'query': 'normalized_event1',
                'old_latter': 'old_normalized_event2',
                'latter': 'normalized_event2',
                'bpn': 'num_base_phrases2',
                'latter_word_count': 'num_morphemes2',
                'former_words': 'content_words1',
                'latter_words': 'content_words2',
                'basic_event': 'core_event_pair',
            },
            inplace=True
        )
    elif args.v == 3:
        df.rename(
            columns={
                'sent': 'orig',
                'sep_fmr': 'morphemes1',
                'sep_qry': 'normalized_morphemes1',
                'sep_ltr': 'normalized_morphemes2',
                'fmr': 'event1',
                'qry': 'normalized_event1',
                'ltr': 'normalized_event2',
                'n_bp': 'num_base_phrases2',
                'seed': 'core_event_pair',
                'op': 'noisy_prefix',
            },
            inplace=True
        )
    elif args.v == 4:
        df.rename(
            columns={
                'seg_evt1': 'morphemes1',
                'seg_evt1_wo_mod': 'morphemes1_wo_modifier',
                'seg_norm_evt1': 'normalized_morphemes1',
                'seg_norm_evt2': 'normalized_morphemes2',
                'evt1': 'event1',
                'norm_evt1': 'normalized_event1',
                'norm_evt2': 'normalized_event2',
                'num_bps2': 'num_base_phrases2',
                'num_mrphs2': 'num_morphemes2',
                'cws1': 'content_words1',
                'ya_cws1': 'ya_content_words1',
                'cws2': 'content_words2',
                'ya_cws2': 'ya_content_words2',
                'core_ep': 'core_event_pair',
                'neg1': 'negation1',
                'neg2': 'negation2',
                'pot1': 'potential1',
                'pot2': 'potential2',
                'norm_pred1': 'normalized_predicate1',
                'norm_pred2': 'normalized_predicate2',
                'seg_cbep': 'segmented_cbep',
            },
            inplace=True
        )
    else:
        raise ValueError('unknown version')
    columns = [column for column in COLUMNS if column in df.columns]
    df = df.reindex(columns=columns)
    df.to_csv(args.OUTPUT, index=False, sep='\t')


if __name__ == '__main__':
    main()
