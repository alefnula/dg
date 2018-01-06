__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import abc
import pandas as pd
from dg.enums import Mode
from dg.config import Config
from tea.utils import get_object
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

    def input_fn(self, filename, mode):
        """Input function that transforms the dataset file into the format
        needed by the model.

        Default implementation just reads the csv file using pandas, and
        returns the X and y tensors.

        Args:
            filename (str): Path to the filename
            mode (Mode):
        Returns:
            X: Dataset features
            y: Dataset targets
        """
        data = pd.read_csv(filename)
        X = data[self.config.features]
        if self.config.targets in (None, [], tuple()):
            y = None
        else:
            y = data[self.config.targets]
        return X, y

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
        X, y = self.input_fn(train_set, Mode.TRAIN)
        return self.fit(X, y)

    # Optional
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

        This function receives a path to the dataset, reads the files from disk
        and predicts values.

        Args:
            dataset (str): Path to the dataset csv
        """
        X, _ = self.input_fn(dataset, Mode.PREDICT)
        return self.predict(X)

    # Optional
    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Args:
            X (array-like, shape = [n_samples, n_features]):
                Input samples.

        Returns:
            C (array-like, shape = [n_samples, n_classes]):
                Returns the probability of the samples for each class in the
                model. The columns correspond to the classes in sorted order,
                as they appear in the training dataset.
        """

    def predict_proba_dataset(self, dataset):
        """Default implementation of the predict proba dataset function.

        This function receives a path to the dataset, reads the files from disk
        and predicts values.

        Args:
            dataset (str): Path to the dataset csv
        """
        X, _ = self.input_fn(dataset, Mode.PREDICT)
        return self.predict_proba(X)

    # Optional
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
        X, _ = self.input_fn(dataset, Mode.TRANSFORM)
        return self.transform(X)

    def score(self, X, y, sample_weight=None, metrics=None):
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
            metrics: Optional dictionary of metrics functions to use instead of
                the selected ones for regression and classification.

        Returns:
            float: For classification: mean accuracy of self.predict(X) wrt. y.
                   For regression: R^2 of self.predict(X) wrt. y.
        """
        # If metrics dictionary is passed in use it to calculate the metrics
        if metrics is not None:
            return {
                key: get_object(value)(y, self.predict(X))
                for key, value in metrics.items()
            }

        # If metrics are None try to get the scoring function from the
        # configuration file
        proba = self.conrig.get('metrics.proba', False)
        score = self.config.get('metrics.score', None)
        predict_func = self.predict_proba if proba else self.predict

        if score is not None:
            return get_object(score)(y, predict_func(X),
                                     sample_weight=sample_weight)

        # Finally try the default estimators for classification and regression
        estimator_type = getattr(self, '_estimator_type', None)
        if estimator_type == 'classifier':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, predict_func(X),
                                  sample_weight=sample_weight)
        elif estimator_type == 'regressor':
            from sklearn.metrics import r2_score
            return r2_score(y, predict_func(X),
                            sample_weight=sample_weight,
                            multioutput='variance_weighted')
        else:
            # Don't know how to score the model, just return 0
            return 0

    def score_dataset(self, dataset, sample_weight=None, metrics=None):
        """Default implementation of the score dataset function.

        This function receives a path to the dataset and target labels, reads
        the files from disk and calcualate score.

        Args:
            dataset (str): Path to the dataset
            sample_weight (array-like, shape = [n_samples], optional):
                Sample weights.
            metrics: Optional dictionary of metrics functions to use instead of
                the selected ones for regression and classification.
        """
        X, y = self.input_fn(dataset, Mode.EVAL)
        return self.score(X, y, sample_weight, metrics)
