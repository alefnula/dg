__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 18 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import dg
import os


@dg.command
def board():
    """Run TensorBoard in the tensorflow models subdirectory"""
    os.system('tensorboard --logdir models/tensorflow')


@dg.command
def notebook():
    """Run Jupyter notebook in the notebooks subdirectory"""
    os.chdir('notebooks')
    os.system('jupyter notebook')
