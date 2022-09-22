import json
import os
from argparse import ArgumentParser
from importlib import import_module

from utils import ObjectHook


def main():
    parser = ArgumentParser()
    parser.add_argument('CONFIG', type=str, help='path to config file')
    parser.add_argument('--gpu', default=None, type=str, help='gpu numbers')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--seed', default=0, type=int, help='seed value')
    parser.add_argument('--dump', action='store_true', help='for error analysis')
    args = parser.parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    with open(args.CONFIG, mode='r') as f:
        cfg = json.load(f, object_hook=ObjectHook)

    tester = getattr(import_module(cfg.tester.module), cfg.tester.class_)(args, cfg)
    if args.split == 'dev':
        for corpus, data_loader in tester.dev_data_loader.items():
            ret = tester.evaluation_loop(data_loader, do_error_analysis=args.dump)
            if args.dump and args.local_rank == 0:
                print(f'*** dump error analysis in {tester.load_dir} ***')
                with tester.load_dir.joinpath('error_analysis.jsonl').open(mode='w') as f:
                    for line in ret:
                        json.dump(line, f, ensure_ascii=False)
                        f.write('\n')
    else:
        for corpus, data_loader in tester.test_data_loader.items():
            tester.evaluation_loop(data_loader)
        if args.dump:
            print('ignore dump option')


if __name__ == '__main__':
    main()
