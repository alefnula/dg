__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import abc
from dg.model.model import Model


class SklearnModel(Model, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.model_ = None

    @abc.abstractmethod
    def model_fn(self):
        """Function that creates a tensorflow model.
        """

    def fit(self, X, y=None):
        self.model_ = self.model_fn()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def transform(self, X):
        return self.model_.transform(X)
