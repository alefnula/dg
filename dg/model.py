__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import os
import io
import abc
import yaml
import types
import pandas as pd
from tea import shell
from tea.utils import get_object
from dg.config import Config
from sklearn.base import BaseEstimator, TransformerMixin


class Model(BaseEstimator, TransformerMixin):
    """Base class for all estimators

        Attributes:
            config (Config): Configuration instance
        """

    id = None
    'Estimator id'

    _estimator_type = None

    def __init__(self):
        self.config = Config()

    def __str__(self):
        return self.id

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

    def score(self, X, y, sample_weight=None):
        """Scores the model.

        If scoring function is defined in the configuration file, this function
        will use that scoring function, else:

        For classification:

        Returns the mean accuracy on the given test data
        and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.


        For regression:

        Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Args:
            X (array-like, shape = (n_samples, n_features)): Test samples.
            y (array-like, shape = (n_samples) or (n_samples, n_outputs)):
                True labels for X.
            sample_weight (array-like, shape = [n_samples], optional):
                Sample weights.

        Returns:
            float: For classification: mean accuracy of self.predict(X) wrt. y.
                   For regression: R^2 of self.predict(X) wrt. y.
        """
        score = self.config.get('metrics.score', None)
        if score is not None:
            score = get_object(score)
            return score(y, self.predict(X), sample_weight=sample_weight)
        else:
            if self._estimator_type == 'classifier':
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, self.predict(X),
                                      sample_weight=sample_weight)
            elif self._estimator_type == 'regressor':
                from sklearn.metrics import r2_score
                return r2_score(y, self.predict(X),
                                sample_weight=sample_weight,
                                multioutput='variance_weighted')
            else:
                # Don't know how to score the model, just return 0
                return 0


class Classifier(Model, metaclass=abc.ABCMeta):
    """Base class for all classifiers"""

    _estimator_type = 'classifier'


class Regressor(Model, metaclass=abc.ABCMeta):
    """Base class for all regressors"""

    _estimator_type = 'regressor'


def conform_sklearn_to_model(id, obj):
    """If the object is one of the scikit learn estimators, we add methods
    needed to work with the data.
    """
    obj.id = id
    obj.config = Config()
    obj.__str__ = types.MethodType(Model.__str__, obj)
    obj.fit_dataset = types.MethodType(Model.fit_dataset, obj)
    obj.predict_dataset = types.MethodType(Model.predict_dataset, obj)
    obj.transform_dataset = types.MethodType(Model.transform_dataset, obj)
    obj.score = types.MethodType(Model.score, obj)
    if not hasattr(obj, '_estimator_type'):
        obj._estimator_type = None
    return obj


def strip_model_function(obj):
    del obj.config
    del obj.__str__
    del obj.fit_dataset
    del obj.predict_dataset
    del obj.transform_dataset
    del obj.score
    return obj


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
        """Saves the tensorflow model.

        Args:
            model_dir (str): Path to the directory where the model should be
                saved.
        """
        shell.gcopy(os.path.join(self.model_dir, '*'), f'{model_dir}/')
        with io.open(os.path.join(model_dir, 'params.yaml')) as f:
            yaml.safe_dump(self.get_params(), f)

    @classmethod
    def load(cls, model_dir):
        """Load the production model

        Args:
            model_dir (str): Path to the model dir from where we should load
                the model.
        """
        config = Config()
        with io.open(os.path.join(model_dir, 'params.yaml')) as f:
            params = yaml.safe_load(f)
        model = cls(**params)
        model.model_dir = config.get_model_dir(tensorflow=True)
        model.estimator = model._create_estimator(model_dir)
        return model
