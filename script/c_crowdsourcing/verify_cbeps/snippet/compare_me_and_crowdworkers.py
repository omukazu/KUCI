from argparse import ArgumentParser
from pprint import pprint


def main():
    parser = ArgumentParser()
    parser.add_argument('a2b', type=str, help='path to manual evaluation result of a2b')
    parser.add_argument('other', type=str, help='path to manual evaluation result of other')
    args = parser.parse_args()

    conf_mtx = [[0, 0], [0, 0]]

    for crowdworkers, input_path in enumerate([args.a2b, args.other]):
        with open(input_path, mode='r') as f:
            lines = [line.strip().split('\t') for line in f]
            for event_pair, _ in lines:
                myself = crowdworkers ^ event_pair.startswith('*')
                conf_mtx[myself][crowdworkers] += 1

    pprint(conf_mtx, width=10)


if __name__ == '__main__':
    main()
