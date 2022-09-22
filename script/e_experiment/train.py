import json
import os
from argparse import ArgumentParser
from importlib import import_module
from math import ceil

import torch.distributed as dist
from transformers import logging

from utils import ObjectHook


logging.set_verbosity_error()


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('--gpu', default=None, type=str, help='gpu numbers')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--seed', default=0, type=int, help='seed value')
    parser.add_argument('--dry-run', action='store_true', help='whether to validate first')
    args = parser.parse_args()
    # torchrun
    args.local_rank = int(os.environ["LOCAL_RANK"])

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    trainer = getattr(import_module(cfg.trainer.module), cfg.trainer.class_)(args, cfg)

    if args.dry_run:
        for corpus, data_loader in trainer.dev_data_loader.items():
            print(f'*** evaluate on {corpus} ***')
            trainer.evaluation_loop(data_loader, -1, snap=False)

    if cfg.hparams.epoch:
        num_epochs = cfg.hparams.epoch
    else:
        num_iter_per_epoch = len(trainer.train_data_loader)
        num_step_per_epoch = ceil(num_iter_per_epoch / cfg.hparams.accumulation_steps)
        num_epochs = ceil(cfg.hparams.num_training_steps / num_step_per_epoch)

    for epoch in range(num_epochs):
        if cfg.hparams.num_training_steps and trainer.training_steps >= cfg.hparams.num_training_steps:
            break

        if args.local_rank == 0:
            trainer.logger.info(
                f'*** {args.fold}-{args.seed}-{epoch + 1} ({trainer.best["score"]:.03f}) ***'
            )

        trainer.training_loop(epoch)
        if cfg.trainer.preference.evaluate_on_train:
            trainer.evaluate('train', epoch)

        if (
            args.local_rank == 0 and
            cfg.trainer.preference.save_per_epoch and
            (epoch + 1) % cfg.trainer.preference.save_per_epoch == 0
        ):
            trainer.save(ext=f'.{epoch + 1}')
        dist.barrier()

        if cfg.dataset.class_ == 'AMLM':
            continue

        for i, (corpus, data_loader) in enumerate(trainer.dev_data_loader.items()):
            score = trainer.evaluation_loop(data_loader, epoch)
            if i == 0:  # assume the number of evaluation corpora is 1
                if score > trainer.best['score']:
                    if args.local_rank == 0:
                        print('update best score')
                    trainer.best.update({
                        'epoch': epoch,
                        'score': score,
                        'patience': 0
                    })
                    trainer.store_state_dict()
                else:
                    trainer.best['patience'] += 1

        if cfg.hparams.patience and trainer.best['patience'] >= cfg.hparams.patience:
            break

    if 'test' in cfg.dataset.split.keys():
        trainer.load('best')
        dist.barrier()
        for corpus, data_loader in trainer.test_data_loader.items():
            trainer.evaluation_loop(data_loader, trainer.best['epoch'])

    if args.local_rank == 0:
        trainer.save()
        trainer.watcher.dump()


if __name__ == '__main__':
    main()
