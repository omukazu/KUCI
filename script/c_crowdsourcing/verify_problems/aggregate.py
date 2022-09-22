import json
from argparse import ArgumentParser
from collections import defaultdict as ddict
from pathlib import Path

from utils import get_logger, set_file_handlers


L = get_logger(__name__)


def yield_qa(f) -> tuple[str, str]:
    for line in f:
        problem = json.loads(line)
        context = ''.join(problem['context'].split(' '))
        choices = tuple(
            map(lambda x: ''.join(problem[x].split(' ')), [key for key in problem.keys() if key.startswith('choice')])
        )
        key = f'{context}@{"@".join(choices)}'
        correct_choice = choices[ord(problem['label']) - 97]
        # key: string of context and choices connected with "@" e.g. お腹が空いたので@コーヒーを飲む@ご飯を食べる@眠くなる@汗をかく
        yield key, correct_choice


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to result of crowdsourcing')
    parser.add_argument('--problem', type=str, help='path to problems')
    parser.add_argument('--num-choices', default=4, type=int, help='number of choices')
    parser.add_argument('--threshold', default=3, type=int, help='threshold of majority vote')
    args = parser.parse_args()

    input_path = Path(args.INPUT)
    problem_path = Path(args.problems)
    set_file_handlers(L, output_path=input_path.parent.joinpath(problem_path.with_suffix('.log')))

    with open(input_path, mode='r', errors='replace') as f:
        lines = [line.strip().split('\t') for line in f]
        lines = [line for line in lines if line[0][0] not in {'c', 'd'}]
        # convert results according to the specifications of Yahoo! crowdsourcing
        lines = [(f'{line[2].split("####")[1]}@{line[3]}', line[4]) for line in lines]
        # aggregate answers in worker units
        lines = [lines[10 * idx:10 * (idx + 1)] for idx in range(len(lines) // 10)]

    with open(problem_path, mode='r') as f:
        problem2correct_choice = {
            problem: correct_choice for problem, correct_choice in yield_qa(f)
        }

    # store accuracy of individual workers
    aggregated, individuals, lazy_worker_count = ddict(lambda: [0] * args.num_choices), [], 0
    for answers in lines:
        accuracy, indices = 0, []
        for key, answer in answers:
            aggregated[key][key.split('@')[1:].index(answer)] += 1
            indices.append(key.split('@').index(answer))
            if answer == problem2correct_choice[key]:
                accuracy += 1
        else:
            individuals.append(accuracy)
            if len(set(indices)) == 1:
                lazy_worker_count += 1

    count = 0
    for key, distr in aggregated.items():
        # count if the number of votes exceeds the majority
        if distr[key.split('@')[1:].index(problem2correct_choice[key])] >= args.threshold:
            count += 1
        else:
            L.debug(f'incorrect: {key}\n正解: {problem2correct_choice[key]}\n分布: {distr}')

    L.info(f'problems: {len(problem2correct_choice.keys())}')
    L.info(f'lazy worker count: {lazy_worker_count}')
    L.info(f'accuracy of individual crowdworkers: {sum(individuals) / 10 / len(individuals)}')
    L.info(f'accuracy of aggregated answers: {count / len(aggregated)}')


if __name__ == '__main__':
    main()
