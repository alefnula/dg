__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import io
import os
import abc
import glob
import yaml
from tea import shell
from dg.config import Config
from dg.model.model import Model


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

    def fit(self, X, y=None):
        import tensorflow as tf
        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=X, y=y, shuffle=False, num_epochs=1
        )
        self.model_dir = self.config.get_model_dir(tensorflow=True)
        self.estimator = self._create_estimator(self.model_dir)
        experiment = tf.contrib.learn.Experiment(
            estimator=self.estimator,
            train_input_fn=input_fn
        )
        experiment.fit()

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
            experiment.train()

    def predict(self, X):
        import numpy as np
        import tensorflow as tf

        # Predictions is a generator
        predictions = self.estimator.predict(
            input_fn=tf.estimator.inputs.pandas_input_fn(
                x=X, shuffle=False, num_epochs=1
            )
        )
        return np.array(list(predictions))

    def predict_dataset(self, dataset):
        import numpy as np

        predictions = self.estimator.predict(
            input_fn=self.input_fn(dataset)
        )
        return np.array(list(predictions))

    def score_dataset(self, dataset, sample_weight=None, metrics=None):
        e = self.estimator.evaluate(self.input_fn(dataset))
        if metrics is None:
            return e['loss']
        else:
            keys = metrics.keys()
            if hasattr(self, 'METRICS_MAP'):
                return {
                    key: e[self.METRICS_MAP[key]] for key in keys
                }
            else:
                return {
                    key: e.get(key, None) for key in keys
                }

    def save(self, model_dir):
        """Saves the tensorflow model.

        Args:
            model_dir (str): Path to the directory where the model should be
                saved.
        """
        for item in glob.glob(os.path.join(self.model_dir, '*')):
            shell.copy(item, os.path.join(
                model_dir, item.replace(self.model_dir, '', 1).lstrip('/')
            ))
        with io.open(os.path.join(model_dir, 'params.yaml'), 'w') as f:
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
