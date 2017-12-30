__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from dg.utils import bar
from dg import persistence
from dg.metrics import Metrics
import pandas as pd
from tea.utils import get_object


def train_model(model, train_set, eval_set=None, model_dir=None, save=False,
                verbose=False):
    """Train a single model and save it

    Args:
        model: Model to train
        train_set (str): Path to the training dataset
        eval_set (str): Optional path to the evaluation dataset
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
        train_set (str): Path to the training dataset
        eval_set (str): Optional path to the evaluation dataset
        verbose (bool): Print details
    """
    config = dg.Config()
    model_dir = config.get_model_dir()
    if verbose:
        print('Model dir: ', model_dir)
    bar(verbose=verbose)
    for model_id in models:
        model = config.models[model_id].set_params(
            **config.get_params(model_id)
        )
        train_model(
            model, train_set, eval_set, model_dir, save=True, verbose=verbose
        )
        bar(verbose=verbose)


def print_metrics(metrics):
    """Pretty print the metrics"""
    if metrics[Metrics.TRAIN]:
        print('Train:')
        for key, value in metrics[Metrics.TRAIN].items():
            print(f'\t{key}:\t{value:.4f}')

    if metrics[Metrics.EVAL]:
        print('Eval:')
        for key, value in metrics[Metrics.EVAL].items():
            print(f'\t{key}:\t{value:.4f}')

    if metrics[Metrics.TEST]:
        print('Test:')
        for key, value in metrics[Metrics.TEST].items():
            print(f'\t{key}:\t{value:.4f}')


def metrics_to_dict(model, metrics):
    d = {'model': model.id}
    for ds in metrics:
        if metrics[ds] is not None:
            for metric in metrics[ds]:
                d[f'{ds}-{metric}'] = metrics[ds][metric]
    return d


def evaluate_model(model, metrics, verbose=False):
    """Evaluate a single model

    Args:
        model: Model to evaluate
        metrics (dg.metrics.Metrics): Dictionary of metrics objects
        verbose (bool): Print details
    Returns:
        dict: Evaluation metrics
    """
    if verbose:
        print('Evaluating:', model.id)
    db = persistence.Database()
    old_metrics = db.get(model)
    datasets = [ds for ds in metrics.ALL if old_metrics[ds] is None]
    new_metrics = metrics.evaluate(model, datasets=datasets)
    merged_metrics = {}
    for ds in metrics.ALL:
        if old_metrics[ds] is not None:
            merged_metrics[ds] = old_metrics[ds]
        else:
            merged_metrics[ds] = new_metrics[ds]
    if old_metrics != merged_metrics:
        db.add(model, merged_metrics)
    if verbose:
        print_metrics(merged_metrics)
    return metrics_to_dict(model, merged_metrics)


def evaluate(models, train_set=None, eval_set=None, test_set=None,
             verbose=False):
    """Evaluate all models and print out the metrics for evaluation.

    Evaluation is using the production model.

    Args:
        models (list of str): Model names. Pass if you want to evaluate just a
            set of particular models.
        train_set (str): Path to the training dataset
        eval_set (str): Optional path to the evaluation dataset
        test_set (str): Path to the test dataset
        verbose (bool): Print details
    """
    config = dg.Config()
    metrics_class = get_object(config['metrics.class'])
    metrics = metrics_class(train_set, eval_set, test_set)

    all_metrics = []
    bar(verbose=verbose)
    for name in models:
        model = persistence.load(config.models[name])
        all_metrics.append(evaluate_model(model, metrics, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(all_metrics, columns=['model'] + metrics.columns())
    return df


def train_and_evaluate(models, train_set=None, eval_set=None, test_set=None,
                       verbose=False):
    """Train end evaluate models and print out the metrics for evaluation

    Args:
        models (list of str): Model names. Pass if you want to train/evaluate
            just a set of particular models
        train_set (str): Path to the training dataset
        eval_set (str): Optional path to the evaluation dataset
        test_set (str): Path to the test dataset
        verbose (bool): Print details

    """
    config = dg.Config()
    metrics_class = get_object(config['metrics.class'])
    metrics = metrics_class(train_set, eval_set, test_set)
    all_metrics = []
    bar(verbose=verbose)
    for model_id in models:
        model = config.models[model_id].set_params(
            **config.get_params(model_id)
        )
        train_model(model, train_set, eval_set, save=False, verbose=verbose)
        all_metrics.append(evaluate_model(model, metrics, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(all_metrics, columns=['model'] + metrics.columns())
    return df
