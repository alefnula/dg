__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import io
import os
import yaml
import subprocess
from datetime import datetime
from collections import OrderedDict
from dg.exceptions import ConfigNotFound
from tea.utils import get_object, Loader
from tea.dsa.singleton import Singleton


class Config(Singleton):
    """Configuration class

    Attributes:
        project_dir (str): Path to the project root directory
        project_name (str): Name of the project
        data_dir (str): Path to the data directory
        models_dir (str): Path to the models directory
        features (list of str): List of feature columns in the dataset
        targets (list of str): List of target columns in the dataset
        meta (dict): Metadata about the dataset
        datasets (dict): Dictionary of datasets
        functions (dict): Dictionary of functions
    """
    def __init__(self, path=None):
        """
        Args:
            path (str): Path to the configuration file
        """
        #: Data directory
        if path is None:
            # Calculate project directory from the current working dir
            self.project_dir = os.path.abspath(os.getcwd())
            # Get the project name
            self.project_name = os.path.basename(self.project_dir)
            config_file = os.path.join(self.project_dir, 'config',
                                       f'{self.project_name}.yaml')
        else:
            path = os.path.abspath(path)
            # Calculate project directory relative to the config path
            self.project_dir = os.path.dirname(os.path.dirname(path))
            # Get the project name
            self.project_name = os.path.basename(self.project_dir)
            config_file = path

        if not os.path.isfile(config_file):
            raise ConfigNotFound(config_file)

        # Get the data dir
        self.data_dir = os.path.join(self.project_dir, 'data')
        self.models_dir = os.path.join(self.project_dir, 'models')

        with io.open(config_file, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)

        # Setup features and targets
        features = self._data['features']
        if len(features) == 1:
            features = features[0]
        self.features = features
        targets = self._data['targets']
        if len(targets) == 1:
            targets = targets[0]
        self.targets = targets

        self.__models = None
        # Load the meta file
        meta_file = os.path.join(self.data_dir, self['datasets.meta'])
        if os.path.isfile(meta_file):
            with io.open(meta_file) as f:
                self.meta = yaml.safe_load(f)
        else:
            self.meta = {}

        # Load datasets
        self.datasets = {
            dataset: (
                os.path.join(self.data_dir, self[f'datasets.{dataset}'])
                if self[f'datasets.{dataset}'] is not None else None
            )
            for dataset in self.get('datasets', {}).keys()
        }

        # Load functions
        self.functions = self.get('functions', {})

    def __get(self, var):
        current = self._data
        for part in var.split('.'):
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                part = int(part, 10)
                current = current[part]
            else:
                raise KeyError(var)
        return current

    def __getitem__(self, item):
        """Unsafe version, may raise KeyError or IndexError"""
        return self.__get(item)

    def get(self, var, default=None):
        """Safe version which always returns a default value"""
        try:
            return self.__get(var)
        except (KeyError, IndexError):
            return default

    def get_params(self, name):
        """Returns the configuration parameters for the estimator.

        Args:
            name (str): Name of the estimator

        Returns:
            dict: Dictionary of parameters
        """
        return self.get(f'models.{name}', {})

    @property
    def models(self):
        """Return a dictionary of model name to model class mappings.

        Returns:
            dict: {model_name: model_class}
        """
        if self.__models is None:
            from dg.model import Model, conform_sklearn_to_model
            from sklearn.base import BaseEstimator

            loader = Loader()
            loader.load(f'{self.project_name}.models')
            models_dict = {}
            for module in loader.modules.values():
                # First get all non hidden objects
                if hasattr(module, '__all__'):
                    estimators = module.__all__
                else:
                    estimators = [name for name in dir(module)
                                  if not name.startswith('_')]
                # Then filter object by it's type and construct the models
                # dictionary
                for estimator in estimators:
                    obj = getattr(module, estimator)
                    if isinstance(obj, Model):
                        models_dict[obj.id] = obj
                    elif isinstance(obj, BaseEstimator):
                        model_id = getattr(obj, 'id', estimator)
                        models_dict[model_id] = conform_sklearn_to_model(
                            model_id, obj
                        )
                    elif isinstance(obj, type) and issubclass(obj, Model):
                        models_dict[obj.id] = obj()

            self.__models = OrderedDict([
                (model_id, models_dict[model_id])
                for model_id in sorted(models_dict.keys())
            ])
        return self.__models

    def get_model_dir(self, production=False, tensorflow=False):
        """Returns the model dir

        Args:
            production (bool): Select new or production model
            tensorflow (bool): Is this for tensorflow models?

        Returns:
            Model dir is constructed in the following way:
            if production:
                {project_root}/models/production
            else:
                if tensorflow:
                    {project_root}/models/tensorflow/{timestamp}-{git-rev}
                else:
                    {project_root}/models/{timestamp}-{git-rev}

        """
        model_dir = self.models_dir
        if production:
            return os.path.join(model_dir, 'production')

        if tensorflow:
            model_dir = os.path.join(model_dir, 'tensorflow')
        revision = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()
        timestamp = datetime.now().strftime('%Y.%m.%dT%H:%M:%S')

        model_dir = os.path.join(model_dir, f'{timestamp}-{revision}')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir
