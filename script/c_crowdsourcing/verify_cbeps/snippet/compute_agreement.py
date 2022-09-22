import json
from argparse import ArgumentParser

import krippendorff
import numpy as np
import pandas as pd


def compute_fleiss_kappa(rate_list: list) -> float:
    """Return fleiss kappa given aggregate results

    Parameters
    ----------
    rate_list: list
      [size N * k: N = num samples, k = num categories]
    n: int
      num raters

    Return
    ----------
    kappa: float
      fleiss kappa
    """

    N = len(rate_list)
    k = len(rate_list[0])
    n = sum(rate_list[0])
    print(f'num raters: {n}')
    print(f'num samples: {N}')
    print(f'num categories: {k}')

    P_bar = sum([(sum([el**2 for el in row]) - n) / (n * (n - 1)) for row in rate_list]) / N
    Pe_bar = sum([(sum([row[j] for row in rate_list]) / (N * n)) ** 2 for j in range(k)])
    print(f'P_bar: {P_bar}')
    print(f'Pe_bar: {Pe_bar}')

    try:
        kappa = (P_bar - Pe_bar) / (1 - Pe_bar)
    except ZeroDivisionError:
        kappa = float(1)
    return kappa


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    args = parser.parse_args()

    df = pd.read_csv(args.INPUT, sep='\t')
    df['worker_ids'] = df['worker_ids'].map(lambda x: json.loads(x.replace("'", '"')))
    rate_list = [[len(wids[0]), len(wids[1])] for wids in df['worker_ids']]
    kappa = compute_fleiss_kappa(rate_list)
    alpha = krippendorff.alpha(value_counts=np.array(rate_list), level_of_measurement='nominal')
    print(f"Fleiss' kappa: {kappa}")
    print(f"Krippendorff's alpha: {alpha}")


if __name__ == '__main__':
    main()
