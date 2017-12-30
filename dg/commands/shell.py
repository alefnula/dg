__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import dg


@dg.command
def shell():
    """Run IPython shell with loaded configuration and model classes
    """
    from IPython import embed
    user_ns = {}

    config = dg.Config()
    models = {model.id: model for model in config.models.values()}
    user_ns = {
        'config': config,
        'models': models
    }
    embed(user_ns=user_ns)
