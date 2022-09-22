import json
from argparse import ArgumentParser
from pprint import pprint

from utils import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT1', type=str, help='path to input1')
    parser.add_argument('INPUT2', type=str, help='path to input2')
    args = parser.parse_args()

    with open(args.INPUT1, mode='r') as f:
        error_analysis1 = [json.loads(line) for line in tqdm(f)]
    with open(args.INPUT2, mode='r') as f:
        error_analysis2 = [json.loads(line) for line in tqdm(f)]

    conf_mtx = [[[], []], [[], []]]
    for result1, result2 in zip(error_analysis1, error_analysis2):
        # correct -> 0 / incorrect -> 1
        row, column = map(lambda x: int(not x['correct']), [result1, result2])
        context = ''.join(result1.pop('ctx').split(' '))
        choices = list(map(lambda z: ''.join(z.split(' ')), result1.pop('choices')))
        choices = '\n'.join(choices)
        result1.pop('context_tokens')
        result1.pop('choice_tokens')
        result1['problem'] = f'{context}\n{choices}\n'
        conf_mtx[row][column].append(result1)

    num_examples = [[len(column) for column in row] for row in conf_mtx]
    pprint(num_examples, width=max(len(str(row)) for row in num_examples) + 7)
    for example in conf_mtx[1][0]:
        if all(len(string) <= 16 for string in example['problem'].split('\n')):
            print(example)


if __name__ == '__main__':
    main()
