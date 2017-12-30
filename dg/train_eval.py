__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from dg.utils import bar
from dg import persistence
import pandas as pd
from tea.utils import get_object
from collections import OrderedDict


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
        print('Training:', model)
    model.fit_dataset(train_set, eval_set)
    if save:
        if verbose:
            print('Saving:', model)
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
    train_keys = sorted([key for key in metrics if key.startswith('train-')])
    eval_keys = sorted([key for key in metrics if key.startswith('eval-')])
    test_keys = sorted([key for key in metrics if key.startswith('test-')])

    if train_keys:
        print('Train:')
        for key in train_keys:
            print(f'\t{key.split("-")[-1]}:\t{metrics[key]:.4f}')

    if eval_keys:
        print('Eval:')
        for key in eval_keys:
            print(f'\t{key.split("-")[-1]}:\t{metrics[key]:.4f}')

    if test_keys:
        print('Test:')
        for key in test_keys:
            print(f'\t{key.split("-")[-1]}:\t{metrics[key]:.4f}')


def evaluate_model(model, metrics_dict, verbose=False):
    """Evaluate a single model

    Args:
        model: Model to evaluate
        metrics_dict (dict): Dictionary of metrics objects
        verbose (bool): Print details
    Returns:
        dict: Evaluation metrics
    """
    if verbose:
        print('Evaluating:', model)
    all_metrics = []
    for dataset, metrics_obj in metrics_dict.items():
        if metrics_obj is None:
            continue
        if verbose:
            print(f'Evaluating {dataset} set')
        metrics = []
        evaluation = metrics_obj.evaluate(model)
        for key in evaluation:
            metrics.append((f'{dataset}-{key}', evaluation[key]))
        all_metrics.append(metrics)
    metrics = OrderedDict(
        [('model', model.id)] +
        [item for sublist in zip(*all_metrics) for item in sublist]
    )
    if verbose:
        print_metrics(metrics)
    return metrics


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
    metrics = []
    metrics_class = get_object(config['metrics.class'])
    metrics_dict = {
        'train': None if train_set is None else metrics_class(train_set),
        'eval': None if eval_set is None else metrics_class(eval_set),
        'test': None if test_set is None else metrics_class(test_set),
    }

    bar(verbose=verbose)
    for name in models:
        model = persistence.load(config.models[name])
        metrics.append(evaluate_model(model, metrics_dict, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(metrics)
    df.insert(0, 'model', df.pop('model'))
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
    metrics = []
    metrics_class = get_object(config['metrics.class'])
    metrics_dict = {
        'train': None if train_set is None else metrics_class(train_set),
        'eval': None if eval_set is None else metrics_class(eval_set),
        'test': None if test_set is None else metrics_class(test_set),
    }

    bar(verbose=verbose)
    for model_id in models:
        model = config.models[model_id].set_params(
            **config.get_params(model_id)
        )
        train_model(model, train_set, eval_set, save=False, verbose=verbose)
        metrics.append(evaluate_model(model, metrics_dict, verbose=verbose))
        bar(verbose=verbose)

    df = pd.DataFrame(metrics)
    df.insert(0, 'model', df.pop('model'))
    return df
