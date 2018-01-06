__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 09 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

from copy import deepcopy
import dg
from dg.enums import Dataset
from dg.utils import bar, print_and_save_df
from dg.train_eval import train_model, evaluate_model, columns


def create_grid(params, grid_params):
    from sklearn.model_selection import ParameterGrid

    grid_params = deepcopy(grid_params)
    for key, value in params.items():
        if key not in grid_params:
            grid_params[key] = [value]
    return list(ParameterGrid(grid_params))


@dg.command
@dg.argument('-m', '--model', required=True,
             help='Model to train and eval. Default: all models')
@dg.argument('-t', '--test-only', action='store_true',
             help='Evaluate only on test data')
@dg.argument('-o', '--output', help='Path to the output csv file')
@dg.argument('-s', '--silent', action='store_true', help='Don\'t show details')
def grid(model, test_only=False, output=None, silent=False):
    """Implement grid search for model

    Args:
        model (str): Model name for which we want to do a grid search.
        test_only (bool): Evaluate only on test data
        output (str): Path to the output csv file
        silent (bool): Don't print details to standard out.
    """
    config = dg.Config()
    model = config.models[model]
    grid_params = config[f'grid.{model.id}']
    datasets = config.get_datasets(model.id)
    if grid_params is None:
        print('Grid is not defined for this model')
        return

    grid = create_grid(config[f'models.{model}'], grid_params)
    if len(grid) == 0:
        print('Grid is not defined for this model')
        return

    metrics = []
    param_cols = set()
    bar(silent=silent)
    for i, params in enumerate(grid, 1):
        if not silent:
            print(f'{i} out of {len(grid)}')
            print(f'Params: {params}')
        param_cols.update(params.keys())
        model.set_params(**params)
        train_model(model,
                    train_set=datasets[Dataset.TRAIN.value],
                    eval_set=datasets[Dataset.EVAL.value])
        m = evaluate_model(
            model,
            datasets=[Dataset.TEST] if test_only else Dataset.for_eval(),
            silent=silent
        )
        m.update(params)
        metrics.append(m)
        bar(silent=silent)
    import pandas as pd

    cols = ['model'] + sorted(param_cols) + columns()[1:]
    df = pd.DataFrame(metrics, columns=cols)
    print_and_save_df(df, output)
