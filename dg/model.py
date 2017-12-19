__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import os
import abc
from tea import shell
from dg.config import Config
from dg.utils import ensure_dir


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
    def train(self, train_set, eval_set=None):
        """Train the model.

        Args:
            train_set (str): Path to the training dataset
            eval_set (str) Optional path to the evaluation set
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
            features (dict): Dictionary of features
        Returns:
            numpy.ndarray: Array of predicted values
        """
        pass

    def save_model(self, model_dir):
        model_dir = os.path.join(model_dir, self.name)
        ensure_dir(model_dir, directory=True)
        self.save(model_dir)

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


class TensorflowModel(Model, metaclass=abc.ABCMeta):
    """Base Tensorflow Model"""
    def __init__(self, params=None):
        """
        Args:
            params (dict): Dictionary of model parameters
        """
        super().__init__(params)
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

    def __create_estimator(self, model_dir):
        import tensorflow as tf

        tf_params = self.params.copy()
        tf_params.update({
            'model_dir': model_dir,
        })
        config = tf.estimator.RunConfig(
            tf_random_seed=42,
            model_dir=model_dir,
            save_summary_steps=100,
            save_checkpoints_steps=1000,
        )
        return tf.estimator.Estimator(
            model_fn=self.model_fn, config=config, params=tf_params
        )

    def train(self, train_set, eval_set=None):
        import tensorflow as tf

        self.model_dir = self.config.get_model_dir(tensorflow=True)
        self.estimator = self.__create_estimator(self.model_dir)
        experiment = tf.contrib.learn.Experiment(
            estimator=self.estimator,
            train_input_fn=self.input_fn(train_set),
            eval_input_fn=self.input_fn(eval_set) if eval_set else lambda: None
        )
        experiment.train()

    def predict(self, features):
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        # Predictions is a generator
        predictions = self.estimator.predict(
            input_fn=tf.estimator.inputs.pandas_input_fn(
                x=pd.DataFrame(features), shuffle=False, num_epochs=1
            )
        )
        return np.array(list(predictions))

    def save(self, model_dir):
        shell.gcopy(os.path.join(self.model_dir, '*'), f'{model_dir}/')

    def load(self, model_dir=None):
        """Load the production model"""
        save_dir = os.path.join(
            self.config.get_model_dir(production=True), self.name
        ) if model_dir is None else os.path.join(model_dir, self.name)
        self.model_dir = self.config.get_model_dir(tensorflow=True)
        self.estimator = self.__create_estimator(save_dir)
