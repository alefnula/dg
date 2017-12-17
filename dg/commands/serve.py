__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 17 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
from tea.utils import get_object


def serve():
    """Serve models"""
    config = dg.Config()

    server_klass = config.get('server.klass', None)
    if server_klass:
        server = get_object(server_klass)()
    else:
        server = dg.Server()
    server.run()
