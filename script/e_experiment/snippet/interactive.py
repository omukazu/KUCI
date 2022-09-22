import json
import os
import re
from argparse import ArgumentParser
from importlib import import_module

import torch

from utils import ObjectHook, trace_back


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config')
    parser.add_argument('--gpu', default=None, type=str, help='gpu numbers')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--seed', default=0, type=int, help='seed value')
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.split = None

    assert re.match(r'[0-9]|[1-9][0-9]', args.gpu), 'specify single gpu number'

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    tester = getattr(import_module(cfg.tester.module), cfg.tester.class_)(args, cfg)

    with torch.no_grad():
        while True:
            try:
                input_ = input('input sentence >')
                output = tester.interactive(input_)
                print(f'{input_}: {output.tolist()}')
            except Exception as e:
                trace_back(e, print)
                continue


if __name__ == '__main__':
    main()
