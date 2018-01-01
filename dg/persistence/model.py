__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import io
import os
from dg.utils import ensure_dir
from sklearn.externals import joblib


def save(model, model_dir):
    """Save the model

    Args:
        model (dg.models.Model): Model that should be saved
        model_dir (str): Path to the directory where the model should be
            saved.
    """
    model_dir = os.path.join(model_dir, model.id)
    ensure_dir(model_dir, directory=True)
    if hasattr(model, 'save'):
        model.save(model_dir)
    else:
        model_file = os.path.join(model_dir, f'{model.id}.pickle')
        with io.open(model_file, 'wb') as f:
            joblib.dump(model, f)


def load(model, model_dir=None):
    """Load the model

    Args:
        model (dg.models.Model): Model class or instance of the model
        model_dir (str): If `model_dir` is provided, loads the model from
            the model dir, else loads the production model.
    Returns:
        Estimator: Returns the estimator loaded from the save point
    """
    model_dir = model_dir or dg.Config().get_model_dir(production=True)
    model_dir = os.path.join(model_dir, model.id)
    if hasattr(model, 'load'):
        return model.load(model_dir)
    else:
        model_file = os.path.join(model_dir, f'{model.id}.pickle')
        with io.open(model_file, 'rb') as f:
            return joblib.load(f)
