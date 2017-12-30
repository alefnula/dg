__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 30 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import pandas as pd
from tea.utils import get_object
from collections import OrderedDict


class Metrics(object):
    def __init__(self, dataset):
        dataset = pd.read_csv(dataset)
        config = dg.Config()
        self.X = dataset[config.features]
        self.y = dataset[config.targets]
        self.metrics = OrderedDict([
            (metric['name'], get_object(metric['func']))
            for metric in config['metrics.all']
        ])

    def evaluate(self, model):
        prediction = model.predict(self.X)

        return OrderedDict([
            (metric, func(self.y, prediction))
            for metric, func in self.metrics.items()
        ])
