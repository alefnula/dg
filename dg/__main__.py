__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import sys
import argparse
from dg.commands.shell import shell
from dg.commands.create import create
from dg.exceptions import ConfigNotFound


def main():
    parser = argparse.ArgumentParser(prog='dg')
    subparsers = parser.add_subparsers(dest='parser', help='commands')

    subparsers.add_parser(
        'help', help='Print usage information')
    subparsers.add_parser(
        'shell', help=shell.__doc__.splitlines()[0])

    create_parser = subparsers.add_parser(
        'create', help=create.__doc__.splitlines()[0])
    create_parser.add_argument(
        '-p', '--project', help='Project name or path to the project dir')
    create_parser.add_argument(
        '-a', '--author', help='Author\'s full name')
    create_parser.add_argument(
        '-e', '--email', help='Author\'s email address')

    ns, _ = parser.parse_known_args()
    try:
        if ns.parser == 'help':
            parser.print_help()
        elif ns.parser == 'shell':
            shell()
        elif ns.parser == 'create':
            create(ns.project, ns.author, ns.email)
        else:
            parser.print_help()
    except ConfigNotFound as e:
        print(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
