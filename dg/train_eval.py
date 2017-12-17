__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import pandas as pd
from dg.utils import bar


def train_model(model, dataset, model_dir=None, save=False, verbose=False):
    """Train a single model and save it

    Args:
        model: Model to train
        dataset (str): Path to the training dataset
        model_dir (str): Path to the directory where the model should be saved
        save (bool): Save the model
        verbose (bool): Print details
    """
    if verbose:
        print('Training:', model)
    model.train(dataset)
    if save:
        if verbose:
            print('Saving:', model)
        model.save_model(model_dir)


def train(models, dataset, verbose=False):
    """Train all model for production and save them

    Args:
        models (list of str): Model names. Pass if you want to train a just a
            set particular models
        dataset (str): Path to the training dataset
        verbose (bool): Print details
    """
    config = dg.Config()
    model_dir = config.get_model_dir()
    if verbose:
        print('Model dir: ', model_dir)
    bar(verbose=verbose)
    for name in models:
        model = config.models[name]()
        train_model(model, dataset, model_dir, save=True, verbose=verbose)
        bar(verbose=verbose)


def print_metrics(metrics):
    """Pretty print the metrics"""
    trains = sorted([key for key in metrics if key.startswith('train-')])
    tests = sorted([key for key in metrics if key.startswith('test-')])
    print('Train:')
    for key in trains:
        print(f'\t{key.split("-")[-1]}:\t{metrics[key]:.4f}')
    print('Test:')
    for key in tests:
        print(f'\t{key.split("-")[-1]}:\t{metrics[key]:.4f}')


def evaluate_model(model, train_set=None, test_set=None, verbose=False):
    """Evaluate a single model

    Args:
        model: Model to evaluate
        train_set (str): Path to the training dataset
        test_set (str): Path to the test dataset
        verbose (bool): Print details
    Returns:
        dict: Evaluation metrics
    """
    if verbose:
        print('Evaluating:', model)
    metrics = {'model': model.name}
    if train_set is not None:
        train_eval = model.evaluate(train_set)
        for key in train_eval:
            metrics[f'train-{key}'] = train_eval[key]

    if test_set is not None:
        test_eval = model.evaluate(test_set)
        for key in test_eval:
            metrics[f'test-{key}'] = test_eval[key]
    if verbose:
        print_metrics(metrics)
    return metrics


def evaluate(models, train_set=None, test_set=None, verbose=False):
    """Evaluate all models and print out the metrics for evaluation.

    Evaluation is using the production model.

    Args:
        models (list of str): Model names. Pass if you want to evaluate just a
            set of particular models.
        train_set (str): Path to the training dataset
        test_set (str): Path to the test dataset
        verbose (bool): Print details
    """
    config = dg.Config()
    metrics = []
    bar(verbose=verbose)
    for name in models:
        model = config.models[name]()
        model.load()
        metrics.append(evaluate_model(
            model, train_set=train_set, test_set=test_set, verbose=verbose
        ))
        bar(verbose=verbose)

    df = pd.DataFrame(metrics)
    df.insert(0, 'model', df.pop('model'))
    return df


def train_and_evaluate(models, train_set=None, test_set=None, verbose=False):
    """Train end evaluate models and print out the metrics for evaluation

    Args:
        models (list of str): Model names. Pass if you want to train/evaluate
            just a set of particular models
        train_set (str): Path to the training dataset
        test_set (str): Path to the test dataset
        verbose (bool): Print details

    """
    config = dg.Config()
    metrics = []
    bar(verbose=verbose)
    for name in models:
        model = config.models[name]()
        train_model(model, train_set, save=False, verbose=verbose)
        metrics.append(evaluate_model(
            model, train_set=train_set, test_set=test_set, verbose=verbose
        ))
        bar(verbose=verbose)
    df = pd.DataFrame(metrics)
    df.insert(0, 'model', df.pop('model'))
    return df
