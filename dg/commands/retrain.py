__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 13 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from tea.utils import get_object
from dg.train_eval import train
from dg.commands.deploy import deploy


@dg.command
@dg.argument('-v', '--verbose', action='store_true', help='Print details')
def retrain(verbose=False):
    """"Retrain the production model.

    1. Export the database
    2. Train the models
    3. Deploy the models
    4. Send signal to the nextflix to reload new models
    """
    # First export the data
    config = dg.Config()
    export_fn = get_object(config.functions['export'])

    if verbose:
        print('Exporting the data')
    export_fn(config.datasets['export_set'])

    models = list({config['nextflix.nextflix'],
                   config['nextflix.similar_users'],
                   config['nextflix.similar_movies']})
    train(models=models, dataset=config.datasets['export_set'],
          verbose=verbose)
    deploy(models, verbose=verbose)
    # Try to trigger the server to reload models
    if verbose:
        print('Sending signal to the server to reload models')
    try:
        import requests
        requests.post('http://{host}:{port}/reload/'.format(
            host=config['nextflix.host'], port=config['nextflix.port']
        ))
    except:
        pass
