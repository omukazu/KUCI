import gzip
import multiprocessing as mp
import sys
from functools import partial
from logging import Logger, getLogger, INFO, DEBUG, StreamHandler, FileHandler, Formatter
from pathlib import Path
from time import time
from typing import Generator, Optional

from pyknp import KNP, BList
from tqdm import tqdm


tqdm = partial(tqdm, dynamic_ncols=True)


class ObjectHook(dict):
    # return None if key is not found
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    # define __(get|set)state__ for multiprocessing/DistributedDataParallel (pickling the object)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict_):
        self.__dict__ = dict_


class CustomDict(dict):
    def __missing__(self, key):
        return key


def decorator(dst, msg: str):
    def _decorator(func):
        def wrapper(*args, **kwargs):
            dst(f'*** start {msg} ***')
            start = time()
            ret = func(*args, **kwargs)
            dst(f'*** finish {msg} ({int(time() - start)} sec) ***')
            return ret
        return wrapper
    return _decorator


def get_logger(name: str) -> Logger:
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = False
    return logger


def set_file_handlers(logger: Logger, output_path: Optional[Path] = None) -> None:
    formatter = Formatter('[%(asctime)s] (%(levelname)s) %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(INFO)
    logger.addHandler(stream_handler)

    if output_path:
        assert output_path.suffix == '.log', "output_path must end with '.log'"
        try:
            output_path.unlink()
            print('*** delete old log file ***')
        except FileNotFoundError:
            print('*** add new log file ***')

        file_handler = FileHandler(output_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_parsing_results(
    path: str,
    buf_size: int = 0,
    num_examples: int = -1,
    silent: bool = False
) -> Generator[list[BList], None, None]:
    knp = KNP()
    buf, lines = [], ''
    total = 0

    open_ = gzip.open if path.endswith('.gz') else open
    with open_(path, mode='rt', encoding='utf-8', errors='replace') as f:
        bar = f if silent else tqdm(f)
        for line in bar:
            if 0 <= num_examples <= max(len(buf), total):
                break

            if line.startswith('#'):
                line = line.replace(' \n', '\n')
            lines += line

            if line.strip() == 'EOS':
                try:
                    blist = knp.result(lines)
                    buf.append(blist)
                except Exception as e:
                    trace_back(e, print, lines)
                    knp = KNP()

                lines = ''
                if buf_size > 0 and len(buf) % buf_size == 0:
                    total += len(buf)
                    yield buf
                    buf = []
        else:
            yield buf


def multi_processing(
    data,
    func,
    args: Optional[tuple] = None,
    num_jobs: int = 1
) -> list:
    chunk_size = len(data) // num_jobs
    if len(data) % num_jobs > 0:
        chunk_size += 1
    iterable = [
        (data[start: start + chunk_size], *args)
        if args else
        data[start: start + chunk_size]
        for start in range(0, len(data), chunk_size)
    ]
    with mp.Pool(num_jobs) as pool:
        map_ = pool.starmap if args else pool.map
        chunks = map_(func, iterable)
    return chunks


def trace_back(e: Exception, dst, msg: str = '') -> None:
    cls, _, tb = sys.exc_info()
    sfx = f': {msg}' if msg else ''
    dst(f'{cls.__name__}-{e.with_traceback(tb)}{sfx}')
