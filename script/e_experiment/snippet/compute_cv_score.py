import json
from argparse import ArgumentParser
from math import pow, sqrt
from pathlib import Path

from utils import tqdm, trace_back


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input dir')
    parser.add_argument('--num-folds', default=None, type=int, help='number of fold')
    parser.add_argument('--seed', default=None, action='append', help='seeds')
    parser.add_argument('--split', default='test', type=str, help='split name')
    parser.add_argument('--metric', default='acc', choices=['acc', 'prf', 'auc'], help='metric')
    args = parser.parse_args()

    folds = [i + 1 for i in range(args.num_folds)] if args.num_folds > 0 else [0]
    scores = []
    for fold in tqdm(folds):
        seeds = args.seed if args.seed else [None]
        for seed in seeds:
            suffix = f'_{seed}' if seed else ''
            if fold == 0:
                load_dir = Path(args.INPUT + suffix)
            else:
                # confirm fold numbers are 1-origin
                load_dir = Path(args.INPUT + suffix).joinpath(f'fold{fold}')

            try:
                with load_dir.joinpath('snapshot.json').open(mode='r') as f:
                    snapshot = json.load(f)

                if args.metric == 'prf':
                    scores.append(
                        (
                            snapshot[f'{args.split}_prec'][-1][1],
                            snapshot[f'{args.split}_rec'][-1][1],
                            snapshot[f'{args.split}_f1'][-1][1]
                        )
                    )
                else:
                    if args.split == 'dev':
                        _, dev_scores = zip(*snapshot[f'{args.split}_{args.metric}'])
                        scores.append(max(dev_scores))
                    else:
                        scores.append(snapshot[f'{args.split}_{args.metric}'][-1][1])
            except Exception as e:
                trace_back(e, print, str(load_dir))

    if args.metric == 'prf':
        p, r, f = zip(*scores)
        if args.seed and len(args.seed) > 1:
            step = len(args.seed)
            print(f'prec: {p}')
            print(f'rec: {r}')
            print(f'f1: {f}')
            p = [sum(p[offset::step]) / len(p[offset::step]) for offset in range(step)]
            r = [sum(r[offset::step]) / len(r[offset::step]) for offset in range(step)]
            f = [sum(f[offset::step]) / len(f[offset::step]) for offset in range(step)]

        for values in [p, r, f]:
            mean = sum(values) / len(values)
            std = sqrt(sum(pow(value - mean, 2) for value in values) / len(values))
            print(f'cv score: {mean:.03f} ± {std:.03f} ({list(map(lambda x: round(x, 4), values))})')
    else:
        if args.seed and len(args.seed) > 1:
            step = len(args.seed)
            print(f'scores: {scores}')
            scores = [sum(scores[offset::step]) / len(scores[offset::step]) for offset in range(step)]
            # scores = [
            #     sum([score for score in scores[offset::step] if score >= 0.55])
            #     /
            #     len([score for score in scores[offset::step] if score >= 0.55])
            #     for offset in range(step)
            # ]

        mean = sum(scores) / len(scores)
        std = sqrt(sum(pow(score - mean, 2) for score in scores) / len(scores))
        print(f'cv score: {mean:.03f} ± {std:.03f} ({list(map(lambda x: round(x, 4), scores))})')


if __name__ == '__main__':
    main()
