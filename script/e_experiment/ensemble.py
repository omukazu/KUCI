import json
from argparse import ArgumentParser
from collections import defaultdict as ddict
from importlib import import_module
from itertools import combinations
from typing import Generator

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, logging

from utils import tqdm, ObjectHook


logging.set_verbosity_error()


def yield_blending_ratio(num: int) -> Generator[tuple[float], None, None]:
    for dividing_points in combinations(range(1, 100), num - 1):
        ratio = []
        for dividing_point in dividing_points:
            ratio.append(dividing_point - sum(ratio))
        ratio.append(100 - sum(ratio))
        yield tuple(map(lambda x: x / 100., ratio))


def aggregate_metric(results: list[dict[str, ...]], metric: str) -> None:
    if metric == 'f1':
        aggregated = ddict(list)
        for result in results:
            for key, value in result['test_details'].items():
                aggregated[key].append(value)
        for key in ['tp', 'p', 't']:
            aggregated[key] = np.sum(np.array(aggregated[key]), axis=0).tolist()
            print(aggregated[key])
        tp, p, t = map(lambda x: sum(aggregated[x]), ['tp', 'p', 't'])
        prec = tp / p if p > 0 else 0.
        rec = tp / t if t > 0 else 0.
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.
        print(f'f1: {f1:.03f} (prec: {prec:.03f}, rec: {rec:.03f})')
    else:
        raise KeyError('unsupported metric')


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', help='path to config file')
    parser.add_argument('--gpu', default=None, type=str, help='gpu number')
    parser.add_argument('--num-folds', default=0, type=int, help='fold number')
    parser.add_argument('--seed', default=None, action='append', help='seeds')
    parser.add_argument('--metric', default='f1', type=str, help='metric name')
    args = parser.parse_args()

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    folds = [i + 1 for i in range(args.num_folds)] if args.num_folds > 0 else [0]
    results = []
    for fold in folds:
        setattr(args, 'fold', fold)
        ensembler = getattr(import_module(cfg.ensembler.module), cfg.ensembler.class_)(args, cfg)
        for i, data_loader in enumerate(ensembler.dev_data_loader.values()):
            example_ids, outputs, labels = ensembler.get_output(data_loader)
            for ratio in tqdm(yield_blending_ratio(outputs.shape[-1]), leave=False):
                broadcast = torch.tensor(ratio, dtype=outputs.dtype, device=outputs.device).expand(outputs.shape)
                blended = ensembler.ensemble(outputs, ratio=broadcast)
                metrics = ensembler.compute_metrics(data_loader, example_ids, blended, labels)
                if i == 0 and metrics[f'{data_loader.dataset.split}_{args.metric}'] > ensembler.best['score']:
                    ensembler.best.update({
                        'score': metrics[f'{data_loader.dataset.split}_{args.metric}'],
                        'ratio': ratio,
                    })

        for data_loader in ensembler.test_data_loader.values():
            example_ids, outputs, labels = ensembler.get_output(data_loader)
            ratio = ensembler.best['ratio']
            broadcast = torch.tensor(ratio, dtype=outputs.dtype, device=outputs.device).expand(outputs.shape)
            blended = ensembler.ensemble(outputs, ratio=broadcast)
            metrics = ensembler.compute_metrics(data_loader, example_ids, blended, labels)
            print(f'dev blending ratio: {ratio}')
            print(json.dumps(metrics))
            results.append(metrics)

    aggregate_metric(results, args.metric)


if __name__ == '__main__':
    main()
