__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

__all__ = [
    'Model', 'SklearnModel', 'TensorflowModel',
    'ClassifierMixin', 'RegressorMixin'
]

from dg.model.model import Model
from dg.model.sklearn import SklearnModel
from dg.model.tensorflow import TensorflowModel
from dg.model.mixins import ClassifierMixin, RegressorMixin
