__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import dg


@dg.command
def shell():
    """Run IPython shell with loaded configuration and model classes
    """
    from IPython import embed

    config = dg.Config()
    user_ns = {'config': config}
    models = {
        model.__class__.__name__: model
        for model in config.models.values()
    }
    user_ns.update(models)
    embed(user_ns=user_ns)
