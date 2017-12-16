__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import abc
from dg.config import Config


class Model(metaclass=abc.ABCMeta):
    """Base class for all models

    Attributes:
        params (dict): Dictionary of model parameters
        config (Config): Configuration instance
    """

    name = None
    'Model name'

    def __init__(self, params=None):
        """
        Args:
            params (dict): Dictionary of model parameters. If params is not
                provided, values from the configuration file will be used.
        """
        self.config = Config()
        self.params = params or self.config[f'models.{self.name}']

    def __str__(self):
        return self.name

    @abc.abstractmethod
    def train(self, dataset):
        """Train the model.

        Args:
            dataset (str): Path to the training dataset
        """
        pass

    @abc.abstractmethod
    def evaluate(self, dataset):
        """Evaluate the model

        Args:
            dataset (str): Path to the test dataset
        """

    @abc.abstractmethod
    def predict(self, features):
        """Returns prediction for given features.

        Args:
            features (pandas.DataFrame or dict): Data frame or dictionary of
                features
        Returns:
            numpy.ndarray: Array of predicted values
        """
        pass

    @abc.abstractmethod
    def save(self, model_dir):
        """Save the model

        Args:
            model_dir (str): Path to the directory where the model should be
                saved.
        """
        pass

    @abc.abstractmethod
    def load(self, model_dir=None):
        """Load the production model

        Args:
            model_dir (str): If `model_dir` is provided, loads the model from
                the model dir, else loads the production model.
        """
        pass
