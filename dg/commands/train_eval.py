__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from dg import train_eval
from dg.utils import print_and_save_df


def train(models=None, production=False, export=False, verbose=True):
    """Train all model for production and save them

    Args:
        models (list of str): Model names. Pass if you want to train a just a
            set particular models,
        production (bool): Train for production or for evaluation.
        export (bool): Train for production from database export export.
        verbose (bool): Print details
    """
    config = dg.Config()
    dataset = (
        config.datasets.export if export else (
            config.datasets.full_set if production else
            config.datasets.train_set
        )
    )
    models = models or config.models.keys()
    train_eval.train(models, dataset, verbose=verbose)


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
        models, train_set=None if test_only else config.datasets.train_set,
        test_set=config.datasets.test_set, verbose=verbose)
    print_and_save_df(df, output=output)


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
        models, train_set=None if test_only else config.datasets.train_set,
        test_set=config.datasets.test_set, verbose=verbose)
    print_and_save_df(df, output=output)
