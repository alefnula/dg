__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 09 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import pandas as pd
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import dg
from dg.utils import bar, print_and_save_df
from dg.train_eval import train_model, evaluate_model


def create_grid(params, grid_params):
    grid_params = deepcopy(grid_params)
    for key, value in params.items():
        if key not in grid_params:
            grid_params[key] = [value]
    return list(ParameterGrid(grid_params))


def grid(model, test_only=False, output=None, verbose=True):
    """Implement grid search for model

    Args:
        model (str): Model name for which we want to do a grid search.
        test_only (bool): Evaluate only on test data
        output (str): Path to the output csv file
    """
    config = dg.Config()
    grid_params = config[f'grid.{model}']

    if grid_params is None:
        print('Grid is not defined for this model')
        return

    grid = create_grid(config[f'models.{model}'], grid_params)
    if len(grid) == 0:
        print('Grid is not defined for this model')
        return

    train_set = config.datasets.train_set
    test_set = config.datasets.test_set

    metrics = []
    bar(verbose=verbose)
    for i, params in enumerate(grid, 1):
        if verbose:
            print(f'{i} out of {len(grid)}')
            print(f'Params: {params}')
        row = params.copy()
        instance = config.models[model](params)
        train_model(instance, train_set)
        m = evaluate_model(instance,
                           train_set=None if test_only else train_set,
                           test_set=test_set)
        m.pop('model')
        row.update(m)
        metrics.append(row)
        bar(verbose=verbose)

    df = pd.DataFrame(metrics)
    all_columns = set(df.columns.tolist())
    params_columns = set(grid[0].keys())
    metrics_columns = all_columns.difference(params_columns)
    df = df[sorted(params_columns) + sorted(metrics_columns)]
    print_and_save_df(df, output)
