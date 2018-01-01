__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import pandas as pd
from copy import deepcopy
from dg.utils import bar
from dg import persistence
from dg.config import Config
from dg.enums import Mode, Dataset


def train_model(model, train_set, eval_set, model_dir=None, save=False,
                verbose=False):
    """Train a single model and save it

    Args:
        model (dg.Model): Model to train
        train_set (str): Dataset to train on
        eval_set (str): Dataset to use for evaluation during training
        model_dir (str): Path to the directory where the model should be saved
        save (bool): Save the model
        verbose (bool): Print details
    """
    if verbose:
        print('Training:', model.id)
    model.fit_dataset(train_set, eval_set)
    if save:
        if verbose:
            print('Saving:', model.id)
        persistence.save(model, model_dir)


def train(models, train_set, eval_set=None, verbose=False):
    """Train all model for production and save them

    Args:
        models (list of str): Model names. Pass if you want to train a just a
            set particular models
        train_set (dg.enums.Dataset): Dataset to train on
        eval_set (dg.enums.Dataset): Dataset to use for evaluation during
            training.
        verbose (bool): Print details
    """
    config = Config()
    model_dir = config.get_model_dir()
    if verbose:
        print('Model dir: ', model_dir)
    bar(verbose=verbose)
    for model_id in models:
        model = config.models[model_id].set_params(
            **config.get_params(model_id)
        )
        datasets = config.get_datasets(model.id)
        train_model(
            model,
            train_set=datasets[train_set.value],
            eval_set=None if eval_set is None else datasets[eval_set.value],
            model_dir=model_dir, save=True, verbose=verbose
        )
        bar(verbose=verbose)


def print_metrics(metrics):
    """Pretty print the metrics"""
    if metrics[Mode.TRAIN.value]:
        print('Train:')
        for key, value in metrics[Mode.TRAIN.value].items():
            print(f'\t{key}:\t{value:.4f}')

    if metrics[Mode.EVAL.value]:
        print('Eval:')
        for key, value in metrics[Mode.EVAL.value].items():
            print(f'\t{key}:\t{value:.4f}')

    if metrics[Mode.TEST.value]:
        print('Test:')
        for key, value in metrics[Mode.TEST.value].items():
            print(f'\t{key}:\t{value:.4f}')


def metrics_to_dict(model, metrics):
    d = {'model': model.id}
    for ds in metrics:
        if metrics[ds] is not None:
            for metric in metrics[ds]:
                d[f'{ds}-{metric}'] = metrics[ds][metric]
    return d


def columns():
    config = Config()
    cols = ['model']
    for ds in Dataset.for_eval():
        for metric in config.get('metrics.all', {}):
            cols.append(f'{ds.value}-{metric}')
    return cols


def evaluate_model(model, datasets, verbose=False):
    """Evaluate a single model

    Args:
        model (dg.Model): Model to evaluate
        datasets (list of dg.enums.Dataset): List of datasets used for
            evaluation.
        verbose (bool): Print details
    Returns:
        dict: Evaluation metrics
    """
    config = Config()
    metrics = config.get('metrics.all', {})
    if verbose:
        print('Evaluating:', model.id)
    db = persistence.Database()
    old_metrics = db.get(model)
    new_metrics = deepcopy(old_metrics)
    model_datasets = config.get_datasets(model.id)
    for ds in datasets:
        if (
            new_metrics.get(ds.value, None) is None and
            model_datasets[ds.value] is not None
        ):
            new_metrics[ds.value] = model.score_dataset(
                model_datasets[ds.value], metrics=metrics
            )
    if old_metrics != new_metrics:
        db.add(model, new_metrics)
    if verbose:
        print_metrics(new_metrics)
    return metrics_to_dict(model, new_metrics)


def evaluate(models, datasets, verbose=False):
    """Evaluate all models and print out the metrics for evaluation.

    Evaluation is using the production model.

    Args:
        models (list of str): Model names. Pass if you want to evaluate just a
            set of particular models.
        datasets (list of dg.enums.Dataset): List of datasets used for
            evaluation.
        verbose (bool): Print details
    """
    config = Config()
    all_metrics = []
    bar(verbose=verbose)
    for name in models:
        model = persistence.load(config.models[name])
        all_metrics.append(evaluate_model(model, datasets, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(all_metrics, columns=columns())
    return df


def train_and_evaluate(models, datasets, verbose=False):
    """Train end evaluate models and print out the metrics for evaluation

    Args:
        models (list of str): Model names. Pass if you want to train/evaluate
            just a set of particular models
        datasets (list of dg.enums.Dataset): List of datasets used for
            evaluation.
        verbose (bool): Print details

    """
    config = dg.Config()
    all_metrics = []
    bar(verbose=verbose)
    for model_id in models:
        model = config.models[model_id].set_params(
            **config.get_params(model_id)
        )
        dss = config.get_datasets(model.id)
        train_model(model, train_set=dss[Dataset.TRAIN.value],
                    eval_set=dss[Dataset.EVAL.value], save=False,
                    verbose=verbose)
        all_metrics.append(evaluate_model(model, datasets, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(all_metrics, columns=columns())
    return df
