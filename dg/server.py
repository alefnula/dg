__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 17 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
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

        self.models = {
            model: self.config.models[model]()
            for model in self.config['server.models']
        }
        # Load models
        for model in self.models.values():
            model.load()

        # Create server and setup routes
        self.server = Sanic()
        self.server.add_route(self.reload, '/reload/',
                              methods=['POST'])

    # Reload models
    async def reload(self, request):
        """Reload models"""
        for model in self.models.values():
            model.load()
        return json({'message': 'OK'})

    def run(self):
        self.server.run(host=self.config['server.host'],
                        port=self.config['server.port'])
