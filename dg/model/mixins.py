__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'


class ClassifierMixin(object):
    """Base class for all classifiers"""

    _estimator_type = 'classifier'


class RegressorMixin(object):
    """Base class for all regressors"""

    _estimator_type = 'regressor'
