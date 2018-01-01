__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import enum


class Mode(enum.Enum):
    """Standard names for model modes.

    The following standard keys are defined:

    * `TRAIN`: training mode.
    * `EVAL`: evaluation mode.
    * `TEST`: testing mode.s
    * `PREDICT`: inference mode. (unknown targets)
    * `TRANSFORM`: transform mode, for data transformers.
    """

    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'
    PREDICT = 'predict'
    TRANSFORM = 'transform'


class Dataset(enum.Enum):
    """Standard names for datasets.

    The following standard keys are defined:

    * `FULL`: complete non-split dataset
    * `TRAIN`: training mode. (train file)
    * `EVAL`: evaluation mode. (eval file)
    * `TEST`: testing mode. (test file)
    """

    FULL = 'full'
    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'
    PREDICT = 'predict'

    @classmethod
    def for_eval(cls):
        return [cls.TRAIN, cls.EVAL, cls.TEST]
