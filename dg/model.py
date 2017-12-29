__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import os
import io
import abc
import pickle
import pandas as pd
from tea import shell
from dg.config import Config
from dg.utils import ensure_dir, feature_cols
from sklearn.base import (
    BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
)


class Model(BaseEstimator, TransformerMixin):
    """Base class for all estimators

        Attributes:
            config (Config): Configuration instance
        """

    name = None
    'Estimator name'

    def __init__(self):
        self.config = Config()

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Implementation of a fitting function

        Args:
            X (array-like or sparse matrix of shape = [n_samples, n_features]):
                The training input samples.
            y (array-like, shape = [n_samples] or [n_samples, n_outputs]):
                The target values (class labels in classification, real numbers
                in regression).

        Returns:
            Estimator: Returns self.
        """

    def fit_dataset(self, train_set, eval_set=None):
        """Default implementation of the fit dataset function.

        This function receives a path to the train_set and potentially eval_set
        files, and target labels, reads the files from disk and fits the model.

        Args:
            train_set (str): Path to the csv training dataset
            eval_set (str): Path to the csv evaluation set

        Returns:
            Estimator: Returns self.
        """
        train_set = pd.read_csv(train_set)
        X = train_set[self.config.features]
        if self.config.targets in (None, [], tuple()):
            y = None
        else:
            y = train_set[self.config.targets]
        return self.fit(X, y)

    @abc.abstractmethod
    def predict(self, X):
        """Implementation of a predicting function.

        Args:
            X (array-like of shape = [n_samples, n_features]):
                The input samples.

        Returns:
        -------
            array of shape = [n_samples] or [n_samples, n_outputs):
                Predicted values (class labels in classification, real numbers
                in regression).
        """

    def predict_dataset(self, dataset):
        """Default implementation of the predict dataset function.

        This function receives a path to the dataset and target labels, reads
        the files from disk and predicts values.

        Args:
            dataset (str): Path to the dataset csv
        """
        dataset = pd.read_csv(dataset)
        X = dataset[self.config.features]
        return self.predict(X)

    @abc.abstractmethod
    def transform(self, X):
        """Implementation of a transform function.

        Args:
            X (array-like of shape = [n_samples, n_features]):
                The input samples.

        Returns:
            array of int of shape = [n_samples, n_features]: The transofrmed
                array.
        """

    def transform_dataset(self, dataset):
        """Default implementation of the tranform dataset function.

        This function receives a path to the dataset and target labels, reads
        the files from disk and transforms it.

        Args:
            dataset (str): Path to the dataset csv
        """
        dataset = pd.read_csv(dataset)
        X = dataset[self.config.features]
        return self.predict(X)

    def save(self, model_dir):
        """Save the model

        Args:
            model_dir (str): Path to the directory where the model should be
                saved.
        """
        model_dir = os.path.join(model_dir, self.name)
        ensure_dir(model_dir, directory=True)

        model_file = os.path.join(model_dir, f'{self.name}.pickle')
        with io.open(model_file, 'wb') as f:
            pickle.dump(self, f)

        params_file = os.path.join(model_dir, 'params.pickle')
        with io.open(params_file, 'wb') as f:
            pickle.dump(self.get_params(), f)

    @classmethod
    def load(cls, model_dir=None):
        """Load the model

        Args:
            model_dir (str): If `model_dir` is provided, loads the model from
                the model dir, else loads the production model.
        Returns:
            Estimator: Returns the estimator loaded from the save point
        """
        model_dir = model_dir or Config().get_model_dir(production=True)

        model_file = os.path.join(model_dir, cls.name, f'{cls.name}.pickle')
        with io.open(model_file, 'rb') as f:
            model = pickle.load(f)

        params_file = os.path.join(model_dir, cls.name, 'params.pickle')
        with io.open(params_file, 'rb') as f:
            model.set_params(**pickle.load(f))
        return model


class Classifier(Model, ClassifierMixin, metaclass=abc.ABCMeta):
    """Base class for all classifiers"""
    pass


class Regressor(Model, RegressorMixin, metaclass=abc.ABCMeta):
    """Base class for all regressors"""
    pass


# def wrap_sklearn(estimator):
#     """Wraps sklearn objects to conform to dg.Model classes
#
#     Args:
#         estimator (sklearn.base.BaseEstimator): Sklearn estimator we want to
#             wrap and make it conform to the dg.Model
#     """
#     if isinstance(estimator, type):
#         estimator.fit_dataset = Model.fit_dataset
#         estimator.predict_dataset = Model.predict_dataset
#         estimator.save = Model.save
#         estimator.load = partial(Model.load.__func__, estimator)
#     else:
#         estimator.fit_dataset = types.MethodType(
#             Model.fit_dataset, estimator)
#         estimator.predict_dataset = types.MethodType(
#             Model.predict_dataset, estimator)
#         estimator.save = types.ModuleType(Model.save, estimator)
#         estimator.load = partial(Model.load.__func__, estimator.__class__)
#     return estimator


class SklearnModel(Model):
    def __init__(self):
        super().__init__()
        self.model = None  # Implement sklearn model

    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def transform(self, X):
        return self.model.transform(X)


class TensorflowModel(Model, metaclass=abc.ABCMeta):
    """Base Tensorflow Estimator"""
    def __init__(self):
        super().__init__()
        self.model_dir = None
        self.estimator = None

    @abc.abstractmethod
    def input_fn(self, filename):
        """Creates an input function for the provided file

        Args:
            filename (str): Path to the input file
        """

    @abc.abstractmethod
    def model_fn(self, features, labels, mode, params):
        """Function that creates a tensorflow model.

        Args:
             features (tf.Tensor): Features batch tensor
             labels (tf.Tensor): Labels batch tensor
             mode: Training mode
             params (dict): Dictionary of model parameters
        """
        pass

    def _create_estimator(self, model_dir):
        import tensorflow as tf

        tf_params = self.get_params()
        tf_params.update({
            'model_dir': model_dir,
        })
        config = tf.estimator.RunConfig(
            tf_random_seed=42,
            model_dir=model_dir,
            save_summary_steps=tf_params.get('save_summary_steps', 100),
            save_checkpoints_steps=tf_params.get(
                'save_checkpoints_steps', 1000
            ),
        )
        return tf.estimator.Estimator(
            model_fn=self.model_fn, config=config, params=tf_params
        )

    def fit_dataset(self, train_set, eval_set=None, targets=None):
        import tensorflow as tf

        self.model_dir = self.config.get_model_dir(tensorflow=True)
        self.estimator = self._create_estimator(self.model_dir)
        experiment = tf.contrib.learn.Experiment(
            estimator=self.estimator,
            train_input_fn=self.input_fn(train_set),
            eval_input_fn=self.input_fn(eval_set) if eval_set else lambda: None
        )
        if eval_set:
            experiment.train_and_evaluate()
        else:
            experiment.fit()

    def predict(self, features):
        import numpy as np
        import tensorflow as tf

        # Predictions is a generator
        predictions = self.estimator.predict(
            input_fn=tf.estimator.inputs.pandas_input_fn(
                x=features, shuffle=False, num_epochs=1
            )
        )
        return np.array(list(predictions))

    def save(self, model_dir):
        model_dir = os.path.join(model_dir, self.name)
        ensure_dir(model_dir, directory=True)
        shell.gcopy(os.path.join(self.model_dir, '*'), f'{model_dir}/')

    @classmethod
    def load(cls, model_dir=None):
        """Load the production model"""
        config = Config()
        save_dir = os.path.join(
            config.get_model_dir(production=True), cls.name
        ) if model_dir is None else os.path.join(model_dir, cls.name)
        params = config.get_params(cls.name)
        model = cls(**params)
        model.model_dir = config.get_model_dir(tensorflow=True)
        model.estimator = model._create_estimator(save_dir)
        return model
