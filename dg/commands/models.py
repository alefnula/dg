__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 30 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from tea.console.color import cprint
from textwrap import TextWrapper


@dg.command
@dg.argument('-p', '--params', action='store_true',
             help='Print the parameters of the model')
def models(params=False):
    """Lists all models with some additional info"""
    config = dg.Config()
    if len(config.models) == 0:
        return

    longest = max(map(len, config.models.keys()))

    for model_id, model in config.models.items():
        spaces = ' ' * (longest - len(model_id) + 15)
        if model.__doc__ is not None:
            doc = model.__doc__.splitlines()[0]
        else:
            doc = model.__class__.__name__
        cprint(f'{model_id}:{spaces}[blue]{doc}[normal]\n', parse=True)
        if params:
            indent = len(model_id) + len(spaces) + 1
            width = 50 + indent
            wrapper = TextWrapper(
                width=width, initial_indent=' ' * indent,
                subsequent_indent=' ' * indent, break_long_words=False,
                replace_whitespace=True, break_on_hyphens=False
            )

            text = wrapper.fill(', '.join(model.get_params().keys()))
            cprint(f'[cyan]{text}[normal]\n', parse=True)
