__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from dg import train_eval
from dg.utils import print_and_save_df


@dg.command
@dg.argument('-m', '--model', action='append', dest='models',
             help='Models to train. Default: all models')
@dg.argument('-p', '--production', action='store_true',
             help='Train for production not for evaluation')
@dg.argument('-v', '--verbose', action='store_true', help='Print details')
def train(models=None, production=False, verbose=True):
    """Train all model for production and save them

    Args:
        models (list of str): Model names. Pass if you want to train a just a
            set particular models,
        production (bool): Train for production or for evaluation.
        verbose (bool): Print details
    """
    config = dg.Config()
    models = models or config.models.keys()
    train_eval.train(
        models,
        train_set=dg.Dataset.FULL if production else dg.Dataset.TRAIN,
        eval_set=None if production else dg.Dataset.EVAL,
        verbose=verbose
    )


@dg.command
@dg.argument('-m', '--model', action='append', dest='models',
             help='Models to evaluate. Default: all models')
@dg.argument('-t', '--test-only', action='store_true',
             help='Evaluate only on test data')
@dg.argument('-o', '--output', help='Path to the output csv file')
@dg.argument('-v', '--verbose', action='store_true', help='Print details')
def evaluate(models=None, test_only=False, output=None, verbose=False):
    """Evaluate all models and print out the metrics for evaluation.

    Evaluation is using the production model.

    Args:
        models (list of str): Model names. Pass if you want to evaluate just a
            set of particular models.
        test_only (bool): Evaluate only on test data
        output (str): Path to the output csv file
        verbose (bool): Print details
    """
    config = dg.Config()
    models = models or config.models.keys()
    df = train_eval.evaluate(
        models,
        datasets=[dg.Dataset.TEST] if test_only else dg.Dataset.for_eval(),
        verbose=verbose
    )
    print_and_save_df(df, output=output)


@dg.command
@dg.argument('-m', '--model', action='append', dest='models',
             help='Models to train and evaluate. Default: all models')
@dg.argument('-t', '--test-only', action='store_true',
             help='Evaluate only on test data')
@dg.argument('-o', '--output', help='Path to the output csv file')
@dg.argument('-v', '--verbose', action='store_true', help='Print details')
def train_and_evaluate(models=None, test_only=False, output=None,
                       verbose=False):
    """Train end evaluate models and print out the metrics for evaluation

    Args:
        models (list of str): Model names. Pass if you want to train/evaluate
            just a set of particular models
        test_only (bool): Evaluate only on test data
        output (str): Path to the output csv file
        verbose (bool): Print details
    """
    config = dg.Config()
    models = models or config.models.keys()
    df = train_eval.train_and_evaluate(
        models,
        datasets=[dg.Dataset.TEST] if test_only else dg.Dataset.for_eval(),
        verbose=verbose
    )
    print_and_save_df(df, output=output)
