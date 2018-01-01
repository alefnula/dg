__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 01 January 2018'
__copyright__ = 'Copyright (c)  2018 Viktor Kerkez'

import dg
from dg.utils import print_and_save_df
import pandas as pd
from collections import OrderedDict
from dg.persistence import Database


@dg.command
@dg.argument('-m', '--model', action='append', help='Model id')
@dg.argument('-p', '--params', action='store_true', help='Show params')
@dg.argument('-s', '--sort', help='Metrics key on which to sort')
@dg.argument('-d', '--descending', action='store_true',
             help='Sort in descending order')
@dg.argument('-o', '--output', help='Output file')
def metrics(model, params=False, sort=None, descending=False, output=None):
    """Show metrics from the metrics database

    Args:
        model (list of str): Model ids
        params (bool): Show model parameters if True
        sort (str): Column name on which to sort
        descending (bool): Sort in descending order
    """
    if params and len(model) > 1:
        print('Params can be shown only for one model')
        return
    db = Database()
    all = OrderedDict()
    for model_id, model_params, model_metrics in db.metrics(model):
        all.setdefault('model', []).append(model_id)
        if params:
            for param, value in model_params.items():
                all.setdefault(param, []).append(value)
        for key, metrics_data in model_metrics.items():
            if metrics_data is not None:
                for m, value in metrics_data.items():
                    all.setdefault(f'{key}-{m}', []).append(value)
    df = pd.DataFrame(all)
    if sort:
        df.sort_values(sort, ascending=not descending, inplace=True)
    print_and_save_df(df, output=output)
