import json
from argparse import ArgumentParser
from math import log10

import numpy as np
import pandas as pd

from utils import ObjectHook


def main():
    parser = ArgumentParser()
    parser.add_argument('INPUT', type=str, help='path to input')
    parser.add_argument('OUTPUT', type=str, help='path to output')
    args = parser.parse_args()

    with open(args.INPUT, mode='r') as f:
        input_ = json.load(f, object_hook=ObjectHook)

    dfs = []
    for result in input_.results:
        df = pd.DataFrame(
            [[int(num_examples), accuracy] for num_examples, accuracy in result.accuracy.items()],
            columns=['model', 'num_examples', 'accuracy']
        )
        df['model'] = result.model
        df['human'] = input_.human_accuracy
        df['logX'] = df['num_examples'].map(lambda x: log10(x))

        x_mu = sum(df['logX'].values) / len(df)
        y_mu = sum(df['accuracy'].values) / len(df)
        x_var = sum([pow(logx - x_mu, 2) for logx in df['logX']]) / len(df)
        xy_var = sum([logx * y for logx, y in df[['logX', 'accuracy']].values]) / len(df) - x_mu * y_mu

        a = xy_var / x_var
        b = y_mu - a * x_mu

        exponent = round(log10(input_.upper_bound))
        x_start = 500
        mtx = np.array(
            [list(map(lambda x: f'{x:.3f}', [a * log10(x_start) + b, a * exponent + b])) for _ in range(len(df))]
        )

        right_join = pd.DataFrame(mtx, columns=['x', 'y'])
        df = pd.concat([df, right_join], axis=1)
        df.drop(['logX'], axis=1, inplace=True)
        dfs.append(df)

        print(
            f'number of examples necessary to achieve human performance: '
            f'{10 ** ((input_.human_accuracy - b) / a):01f} ({result.model})'
        )
        print(10 ** ((1.0 - b) / a))

    df = pd.concat(dfs)
    df.to_csv(args.OUTPUT, sep='\t', index=False)


if __name__ == '__main__':
    main()

