__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 17 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from dg import persistence
from sanic import Sanic
from sanic.response import json


class Server(object):
    """Base class for all server object

    Attr:
        config (dg.Config): Instance of the configuration object
        models (dict): Dictionary of {model_name: model_instance} mappings
        server (sanic.Server): Instance of the Sanic server for adding routes
            in the subclasses
    """
    def __init__(self):
        self.config = dg.Config()

        # Load models
        self.models = {
            model: persistence.load(self.config.models[model])
            for model in self.config['server.models']
        }

        # Create server and setup routes
        self.server = Sanic()
        self.server.add_route(self.reload, '/reload/',
                              methods=['POST'])

    # Reload models
    async def reload(self, request):
        """Reload models"""
        for name, model in self.models:
            self.models[name] = persistence.load(model)
        return json({'message': 'OK'})

    def run(self):
        self.server.run(host=self.config['server.host'],
                        port=self.config['server.port'])
