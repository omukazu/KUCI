from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    args = parser.parse_args()

    counts = [0, 0, 0]
    for input_path in Path(args.INPUT).glob('**/*.count'):
        with open(input_path, mode='r') as f:
            lines = [int(line.strip()) for line in f]
            for idx, line in enumerate(lines):
                counts[idx] += line

    (
        num_contingent_event_pairs,
        num_reliable_contingent_event_pairs,
        num_contingent_basic_event_pairs
    ) = counts
    print(f'number of contingent event pairs: {num_contingent_event_pairs}')
    print(f'number of reliable contingent event pairs: {num_reliable_contingent_event_pairs}')
    print(f'number of contingent basic event pairs: {num_contingent_basic_event_pairs}')


if __name__ == '__main__':
    main()
