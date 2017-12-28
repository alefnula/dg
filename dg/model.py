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


class Model(BaseEstimator):
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

    def fit_dataset(self, train_set, eval_set=None, targets=None):
        """Default implementation of the fit dataset function.

        This function receives a path to the train_set and potentially eval_set
        files, and target labels, reads the files from disk and fits the model.

        Args:
            train_set (str): Path to the csv training dataset
            eval_set (str): Path to the csv evaluation set
            targets: (list of str): List of target columns

        Returns:
            Estimator: Returns self.
        """
        train_set = pd.read_csv(train_set)
        if targets is not None:
            features = feature_cols(train_set.columns, targets)
            X = train_set[features]
            y = train_set[targets]
        else:
            X = train_set
            y = None
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

    def predict_dataset(self, dataset, targets=None):
        """Default implementation of the predict dataset function.

        This function receives a path to the dataset and target labels, reads
        the files from disk and predicts values.

        Args:
            dataset (str): Path to the dataset csv
            targets (list of str): List of target columns
        """
        dataset = pd.read_csv(dataset)
        if targets is not None:
            features = feature_cols(dataset.columns, targets)
            X = dataset[features]
        else:
            X = dataset
        return self.predict(X)

    def save(self, model_dir):
        """Save the model

        Args:
            model_dir (str): Path to the directory where the model should be
                saved.
        """
        model_dir = os.path.join(model_dir, self.name)
        ensure_dir(model_dir, directory=True)
        pickle_file = os.path.join(model_dir, f'{self.name}.pickle')
        with io.open(pickle_file, 'wb') as f:
            pickle.dump(self, f)

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
        pickle_file = os.path.join(model_dir, cls.name, f'{cls.name}.pickle')

        with io.open(pickle_file, 'rb') as f:
            return pickle.load(f)


class Classifier(Model, ClassifierMixin, metaclass=abc.ABCMeta):
    """Base class for all classifiers"""
    pass


class Regressor(Model, RegressorMixin, metaclass=abc.ABCMeta):
    """Base class for all regressors"""
    pass


class Transformer(Model, TransformerMixin, metaclass=abc.ABCMeta):
    """Base class for all transformers"""

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
