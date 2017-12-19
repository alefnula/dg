__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import os
import sys
import logging
import argparse
from tea.utils import Loader
from dg.config import Config
from dg.command import Command
from dg.exceptions import ConfigNotFound


logger = logging.getLogger(__name__)


def main():
    # For some reason current working directory is not in the python path
    # when dg is installed with pip
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    parser = argparse.ArgumentParser(prog='dg')
    subparsers = parser.add_subparsers(dest='parser', help='commands')

    subparsers.add_parser(
        'help', help='Print usage information')

    loader = Loader()
    loader.load('dg.commands')
    # If we are in the project configuration file will exist and then we can
    # load the project commands
    try:
        config = Config()
        loader.load(f'{config.project_name}.commands')
    except ConfigNotFound:
        pass

    # Print out the modules it could not load
    for module, error in loader.errors.items():
        logger.warning('Could not load "%s": %s', module, error)

    commands = Command.get_instances()

    for command in sorted(commands.values()):
        command.create_subparser(subparsers)

    ns, _ = parser.parse_known_args()
    try:
        if ns.parser == 'help':
            parser.print_help()
        elif ns.parser in commands:
            command = commands[ns.parser]
            command.run(ns)
        else:
            parser.print_help()
    except ConfigNotFound as e:
        print(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
