__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 30 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import pandas as pd
from tea.utils import get_object
from collections import OrderedDict


class Metrics(object):
    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'

    ALL = (TRAIN, EVAL, TEST)

    def __init__(self, train_set=None, eval_set=None, test_set=None):
        config = dg.Config()
        self.train_set = None if train_set is None else pd.read_csv(train_set)
        if self.train_set is not None:
            self.X_train = self.train_set[config.features]
            self.y_train = self.train_set[config.targets]
        self.eval_set = None if eval_set is None else pd.read_csv(eval_set)
        if self.eval_set is not None:
            self.X_eval = self.eval_set[config.features]
            self.y_eval = self.eval_set[config.targets]
        self.test_set = None if test_set is None else pd.read_csv(test_set)
        if self.test_set is not None:
            self.X_test = self.test_set[config.features]
            self.y_test = self.test_set[config.targets]

        self.metrics = OrderedDict([
            (metric['name'], get_object(metric['func']))
            for metric in config['metrics.all']
        ])

    def columns(self):
        columns = []
        for ds in self.ALL:
            for metric in self.metrics:
                columns.append(f'{ds}-{metric}')
        return columns

    def evaluate(self, model, datasets=None):
        evaluation = {
            self.TRAIN: None,
            self.EVAL: None,
            self.TEST: None
        }
        datasets = datasets or []
        if self.TRAIN in datasets and self.train_set is not None:
            prediction = model.predict(self.X_train)
            evaluation['train'] = {
                metric: func(self.y_train, prediction)
                for metric, func in self.metrics.items()
            }

        if self.EVAL in datasets and self.eval_set is not None:
            prediction = model.predict(self.X_eval)
            evaluation['eval'] = {
                metric: func(self.y_eval, prediction)
                for metric, func in self.metrics.items()
            }

        if self.TEST in datasets and self.test_set is not None:
            prediction = model.predict(self.X_test)
            evaluation['test'] = {
                metric: func(self.y_test, prediction)
                for metric, func in self.metrics.items()
            }
        return evaluation
