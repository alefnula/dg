__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 06 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import os
import dg
import glob
import shutil
from datetime import datetime
from dg.utils import ensure_dir, bar


@dg.command
@dg.argument('-m', '--model', action='append', dest='models',
             help='Models do deploy. Default: All found models')
@dg.argument('-v', '--verbose', action='store_true', help='Print details')
def deploy(models=None, verbose=False):
    """Deploy the latest model to production

    Args:
        models (list of str): Names of the models we want to deploy
        verbose (bool): Print details
    """
    config = dg.Config()
    production_dir = config.get_model_dir(production=True)
    models_dir = os.path.dirname(production_dir)

    models = models or config.models.keys()

    files = [
        os.path.basename(x) for x in
        glob.glob(os.path.join(models_dir, '*'))
        # Remove production and tensorflow from the list
        if os.path.basename(x) not in ('production', 'tensorflow')
    ]

    latest = os.path.join(models_dir, sorted(
        files, key=lambda x: datetime.strptime(x[:19], '%Y.%m.%dT%H:%M:%S')
    )[-1])

    ensure_dir(production_dir, directory=True)

    bar(verbose=verbose)
    for model in models:
        if verbose:
            print('Deploying model:', model)
        source = os.path.join(latest, model)
        # If the model is trained in the latest training batch
        if os.path.isdir(source):
            destination = os.path.join(production_dir, model)
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        bar(verbose=verbose)
