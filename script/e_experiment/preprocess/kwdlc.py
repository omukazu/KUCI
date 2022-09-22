import json
import random
from argparse import ArgumentParser
from collections import Counter, defaultdict as ddict
from dataclasses import dataclass, asdict
from pathlib import Path

from pyknp import Juman
from rhoknp import Document, Sentence

from utils import tqdm, get_logger, set_file_handlers


L = get_logger(__name__)
set_file_handlers(L)
J = Juman()

SENSES = (
    '談話関係なし',
    '原因・理由',
    '目的',
    '条件',
    '根拠',
    '対比',
    '逆接'
)
SENSE_MAP = {
    # Labels annotated by experts.
    'その他根拠(逆方向)': '根拠',
    'その他根拠(順方向)': '根拠',
    '原因・理由(逆方向)': '原因・理由',
    '原因・理由(順方向)': '原因・理由',
    '対比(方向なし)': '対比',
    '条件(逆方向)': '条件',
    '条件(順方向)': '条件',
    '目的(逆方向)': '目的',
    '目的(順方向)': '目的',
    '談話関係なし': '談話関係なし',
    '逆接・譲歩(逆方向)': '逆接',
    '逆接・譲歩(順方向)': '逆接',
    # Labels annotated by crowd-workers.
    'NIL': '談話関係なし',
    '上記いずれの関係もない': '談話関係なし',
    '原因・理由': '原因・理由',
    '対比': '対比',
    '条件': '条件',
    '根拠': '根拠',
    '目的': '目的',
    '逆接': '逆接'
}


@dataclass
class Example:
    doc_id: str
    clauses: list[str]
    clause1_index: int
    clause2_index: int
    sense: str
    annotator: str
    explicit: bool


def get_clause_start_indices(sentence: Sentence) -> list[int]:
    start_indices = [0]
    for base_phrase in sentence.base_phrases:
        if base_phrase.index != len(sentence.base_phrases) - 1:
            if base_phrase.features.get('節-区切'):
                if base_phrase.features.get('節-区切') in {'連体修飾', '補文'}:
                    continue
                else:
                    start_indices.append(base_phrase.index + 1)
    return start_indices


def get_clauses(sentences: list[Sentence]) -> list[list]:
    clauses = []
    for sentence in sentences:
        start_indices = get_clause_start_indices(sentence)
        clause = []
        for base_phrase in sentence.base_phrases:
            clause.append(base_phrase)
            if base_phrase.index + 1 in start_indices:  # and base_phrase.index != 0:
                clauses.append(clause)
                clause = []
        clauses.append(clause)
    return clauses


def get_surf(clause: list[list]) -> str:
    return ''.join(
        morpheme.surf
        for base_phrase in clause
        for morpheme in base_phrase.morphemes
    )


def read_expert_data(input_path: str, doc_id2document: dict[str, Document]) -> list[list[Example]]:
    doc_id2examples = ddict(list)
    with open(input_path) as f:
        for line in tqdm(f):
            doc_id, clauses, instruction, _, *labels = line.strip().split('\t')

            instruction_parts = instruction.split(' ')
            clause1_index = int(instruction_parts[0]) - 1  # to 0-origin
            clause2_index = int(instruction_parts[2]) - 1  # to 0-origin

            clauses = clauses.split('##')
            assert clauses[clause1_index].startswith('【') and clauses[clause1_index].endswith('】')
            assert clauses[clause2_index].startswith('【') and clauses[clause2_index].endswith('】')

            document = doc_id2document[doc_id]
            clauses_ = get_clauses(document.sentences[:3])
            segmented_clauses = [
                ' '.join(morpheme.midasi for morpheme in J.analysis(get_surf(clause)).mrph_list())
                for clause in clauses_
            ]
            assert all(
                clause.strip('【】') == segmented_clause.replace(' ', '')
                for clause, segmented_clause in zip(clauses, segmented_clauses)
            )

            clause1 = clauses_[clause1_index]
            clause2 = clauses_[clause2_index]
            sense = SENSE_MAP[labels[0]]
            explicit = any(
                key.startswith('節-機能')
                for base_phrase in clause1
                for key in base_phrase.features.keys()
            ) or any(
                key.startswith('節-前向き機能')
                for base_phrase in clause2
                for key in base_phrase.features.keys()
            )
            explicit &= (clause2_index - clause1_index == 1)

            example = Example(
                doc_id=doc_id,
                clauses=segmented_clauses,
                clause1_index=clause1_index,
                clause2_index=clause2_index,
                sense=sense,
                annotator='expert',
                explicit=explicit
            )
            doc_id2examples[example.doc_id].append(example)

    return list(doc_id2examples.values())


def read_crowdsourcing_data(input_path: str, doc_id2document: dict[str, Document]) -> list[list[Example]]:
    doc_id2examples = ddict(list)
    with open(input_path) as f:
        for line in tqdm(f):
            doc_id, clauses, instruction, _, *labels = line.strip().split('\t')

            instruction_parts = instruction.split(' ')
            clause1_index = int(instruction_parts[0]) - 1  # to 0-origin
            clause2_index = int(instruction_parts[2]) - 1  # to 0-origin

            clauses = clauses.split('##')
            assert clauses[clause1_index].startswith('【') and clauses[clause1_index].endswith('】')
            assert clauses[clause2_index].startswith('【') and clauses[clause2_index].endswith('】')

            document = doc_id2document[doc_id]
            clauses_ = get_clauses(document.sentences[:3])
            segmented_clauses = [
                ' '.join(morpheme.midasi for morpheme in J.analysis(get_surf(clause)).mrph_list())
                for clause in clauses_
            ]
            assert all(
                clause.strip('【】') == segmented_clause.replace(' ', '')
                for clause, segmented_clause in zip(clauses, segmented_clauses)
            )

            clause1 = clauses_[clause1_index]
            clause2 = clauses_[clause2_index]
            if labels == ['NIL']:
                sense = SENSE_MAP['NIL']
            else:
                assert len(labels) % 2 == 0
                sense = None
                prob = 0.0
                for i in range(0, len(labels), 2):
                    cur_sense = SENSE_MAP[labels[i]]
                    cur_prob = float(labels[i + 1])
                    if prob < cur_prob:
                        prob = cur_prob
                        sense = cur_sense
                assert sense is not None
            explicit = any(
                key.startswith('節-機能')
                for base_phrase in clause1
                for key in base_phrase.features.keys()
            ) or any(
                key.startswith('節-前向き機能')
                for base_phrase in clause2
                for key in base_phrase.features.keys()
            )
            explicit &= (clause2_index - clause1_index == 1)

            example = Example(
                doc_id=doc_id,
                clauses=segmented_clauses,
                clause1_index=clause1_index,
                clause2_index=clause2_index,
                sense=sense,
                annotator='crowdworker',
                explicit=explicit
            )
            doc_id2examples[example.doc_id].append(example)

    return list(doc_id2examples.values())


def create_k_fold_cross_validation_datasets(
    grouped_examples: list[list[Example]],
    k: int
) -> list[tuple[list[list[Example]], list[list[Example]], list[list[Example]]]]:
    L.debug(f'create {k}-fold cross-validation datasets')

    num_documents = len(grouped_examples)

    indices = list(range(num_documents))
    random.shuffle(indices)

    folds = []
    for i in range(k):  # ドキュメント単位で分割
        test_indices = set(indices[i * num_documents // k:(i + 1) * num_documents // k])
        dev_indices = set(indices[((i + 1) % k) * num_documents // k:(((i + 1) % k) + 1) * num_documents // k])
        train_examples = []
        dev_examples = []
        test_examples = []
        for index, grouped_example in enumerate(grouped_examples):
            if index in test_indices:
                test_examples.append(grouped_example)
            elif index in dev_indices:
                dev_examples.append(grouped_example)
            else:
                train_examples.append(grouped_example)
        folds.append((train_examples, dev_examples, test_examples))
    return folds


def filter_examples_by_document_ids(examples: list[list[Example]], doc_ids: set[str]) -> list[list[Example]]:
    return [grouped_example for grouped_example in examples if grouped_example[0].doc_id not in doc_ids]


def show_stats(grouped_examples: list[list[Example]]) -> None:
    num_documents = len(grouped_examples)
    num_examples = sum(len(grouped_example) for grouped_example in grouped_examples)
    num_labels = Counter([e.sense for grouped_example in grouped_examples for e in grouped_example])
    L.info(f'number of documents: {num_documents}')
    L.info(f'number of examples: {num_examples}')
    L.info(f'number of labels: {num_labels}')


def save_examples(
    output_path: Path,
    examples: list[Example]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode='w') as f:
        f.write('\n'.join(json.dumps(asdict(example), ensure_ascii=False) for example in examples) + '\n')


def preprocess_kwdlc(args) -> None:
    doc_id2knp_text = ddict(str)
    for knp_path in tqdm(Path(args.knp).iterdir()):
        with knp_path.open(mode='r') as f:
            buf = ''
            for line in f:
                buf += line
                if line.strip() == 'EOS':
                    sentence = Sentence.from_knp(buf)
                    doc_id2knp_text[sentence.doc_id] += buf
                    buf = ''
    doc_id2document = {
        doc_id: Document.from_knp(knp_text) for doc_id, knp_text in tqdm(doc_id2knp_text.items())
    }

    ex_examples = read_expert_data(args.expert, doc_id2document)
    cs_examples = read_crowdsourcing_data(args.crowdworker, doc_id2document)

    folds = create_k_fold_cross_validation_datasets(ex_examples, k=5)

    output_root = Path(args.output_root)
    for fold, (expert_train_examples, dev_examples, test_examples) in enumerate(folds):
        L.info(f'fold{fold}')
        L.info('train (expert)')
        show_stats(expert_train_examples)
        L.info('dev (expert)')
        show_stats(dev_examples)
        L.info('test (expert)')
        show_stats(test_examples)

        output_dir = output_root.joinpath('expert').joinpath(f'fold{fold + 1}')
        save_examples(output_dir.joinpath('train.jsonl'), [e for examples in expert_train_examples for e in examples])
        save_examples(output_dir.joinpath('dev.jsonl'), [e for examples in dev_examples for e in examples])
        save_examples(output_dir.joinpath('test.jsonl'), [e for examples in test_examples for e in examples])

        eval_doc_ids = set(grouped_example[0].doc_id for grouped_example in test_examples + dev_examples)
        crowd_train_examples = filter_examples_by_document_ids(cs_examples, eval_doc_ids)

        L.info('train (crowd)')
        show_stats(crowd_train_examples)
        L.info('dev (crowd)')
        show_stats(dev_examples)
        L.info('test (crowd)')
        show_stats(test_examples)

        output_dir = output_root.joinpath('crowd').joinpath(f'fold{fold + 1}')
        save_examples(output_dir.joinpath('train.jsonl'), [e for examples in crowd_train_examples for e in examples])
        save_examples(output_dir.joinpath('dev.jsonl'), [e for examples in dev_examples for e in examples])
        save_examples(output_dir.joinpath('test.jsonl'), [e for examples in test_examples for e in examples])

        output_dir = output_root.joinpath('merged').joinpath(f'fold{fold + 1}')
        intersect_doc_ids = set(
            grouped_example[0].doc_id for grouped_example in expert_train_examples
        )
        intersect_train_examples = filter_examples_by_document_ids(crowd_train_examples, intersect_doc_ids)
        intersect_train_examples += expert_train_examples
        save_examples(output_dir.joinpath('train.jsonl'), [e for examples in intersect_train_examples for e in examples])
        save_examples(output_dir.joinpath('dev.jsonl'), [e for examples in dev_examples for e in examples])
        save_examples(output_dir.joinpath('test.jsonl'), [e for examples in test_examples for e in examples])


def main():
    parser = ArgumentParser()
    parser.add_argument('--expert', type=str, help='path to KWDLC annotated by experts')
    parser.add_argument('--crowdworker', type=str, help='Path to KWDLC annotated by crowdworkers')
    parser.add_argument('--knp', type=str, help='path to KWDLC knp files')
    parser.add_argument('--output_root', type=str, required=True, help='path to output root')
    args = parser.parse_args()

    random.seed(42)

    preprocess_kwdlc(args)


if __name__ == '__main__':
    main()
