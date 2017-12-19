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
from dg.utils import nt
from tea.utils import get_object
from tea.dsa.config import Config as TeaConfig
from tea.dsa.singleton import SingletonMetaclass


class Config(TeaConfig, metaclass=SingletonMetaclass):
    """Configuration class

    Attributes:
        data_dir (str): Path to the data directory
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

        super().__init__(filename=config_file, fmt=Config.YAML,
                         auto_save=False)

        self.__models = None

        # Load the meta file
        meta_file = os.path.join(self.data_dir, self['datasets.meta'])
        if os.path.isfile(meta_file):
            with io.open(meta_file) as f:
                self.meta = nt('Meta', yaml.safe_load(f))
        else:
            self.meta = nt('Meta', {})

        # Load datasets
        self.datasets = nt('Datasets', {
            dataset: os.path.join(self.data_dir, self[f'datasets.{dataset}'])
            for dataset in self.get('datasets', {}).keys()
        })

        # Load functions
        self.functions = nt('Functions', self.get('functions', {}))

    @property
    def models(self):
        """Return a dictionary of model name to model class mappings.

        Returns:
            dict: {model_name: model_class}
        """
        if self.__models is None:
            from dg.model import Model
            module = get_object(f'{self.project_name}.models')
            models = {
                obj.__name__: obj
                for obj in get_object(f'{self.project_name}.models.*')
                if isinstance(obj, type) and issubclass(obj, Model)
            }
            if hasattr(module, '__all__'):
                order = getattr(module, '__all__')
            else:
                order = sorted(models.keys())
            self.__models = OrderedDict([
                (model.name, model) for model in [
                    models[klass_name] for klass_name in order
                ]
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
        model_dir = os.path.join(self.project_dir, 'models')
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

