__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import sys
import argparse
from dg.commands.grid import grid
from dg.commands.shell import shell
from dg.commands.serve import serve
from dg.commands.create import create
from dg.commands.deploy import deploy
from dg.commands.retrain import retrain
from dg.commands.train_eval import train, evaluate, train_and_evaluate
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

    # Training
    train_parser = subparsers.add_parser(
        'train', help=train.__doc__.splitlines()[0])
    train_parser.add_argument(
        '-m', '--model', action='append',
        help='Models to train. Default: all models')
    train_parser.add_argument(
        '-p', '--production', action='store_true',
        help='Train for production not for evaluation')
    train_parser.add_argument(
        '-e', '--export', action='store_true',
        help='Train for production from database export')
    train_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print details')

    # Evaluation
    eval_parser = subparsers.add_parser(
        'eval', help=evaluate.__doc__.splitlines()[0])
    eval_parser.add_argument(
        '-m', '--model', action='append',
        help='Models to evaluate. Default: all models')
    eval_parser.add_argument(
        '-t', '--test-only', action='store_true',
        help='Evaluate only on test data')
    eval_parser.add_argument(
        '-o', '--output', help='Path to the output csv file')
    eval_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print details')

    # Train and evaluate
    teval_parser = subparsers.add_parser(
        'teval', help=train_and_evaluate.__doc__.splitlines()[0])
    teval_parser.add_argument(
        '-m', '--model', action='append',
        help='Models to train and evaluate. Default: all models')
    teval_parser.add_argument(
        '-t', '--test-only', action='store_true',
        help='Evaluate only on test data')
    teval_parser.add_argument(
        '-o', '--output', help='Path to the output csv file')
    teval_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print details')

    # Deploy models
    deploy_parser = subparsers.add_parser(
        'deploy', help=deploy.__doc__.splitlines()[0])
    deploy_parser.add_argument(
        '-m', '--model', action='append',
        help='Models do deploy. Default: All found models')
    deploy_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print details')

    grid_parser = subparsers.add_parser(
        'grid', help=grid.__doc__.splitlines()[0])
    grid_parser.add_argument(
        '-m', '--model', required=True,
        help='Model to train and eval. Default: all models')
    grid_parser.add_argument(
        '-t', '--test-only', action='store_true',
        help='Evaluate only on test data')
    grid_parser.add_argument(
        '-o', '--output', help='Path to the output csv file')
    grid_parser.add_argument(
        '-v', '--verbose', action='store_true', help='Print details')

    # Retrain
    subparsers.add_parser(
        'retrain', help=retrain.__doc__.splitlines()[0])

    # Serve
    subparsers.add_parser(
        'serve', help=serve.__doc__.splitlines()[0])

    ns, _ = parser.parse_known_args()
    try:
        if ns.parser == 'help':
            parser.print_help()
        elif ns.parser == 'shell':
            shell()
        elif ns.parser == 'create':
            create(ns.project, ns.author, ns.email)
        elif ns.parser == 'train':
            train(ns.model, ns.production, ns.verbose)
        elif ns.parser == 'eval':
            evaluate(ns.model, ns.test_only, ns.output, ns.verbose)
        elif ns.parser == 'teval':
            train_and_evaluate(ns.model, ns.test_only, ns.output, ns.verbose)
        elif ns.parser == 'deploy':
            deploy(ns.model, ns.verbose)
        elif ns.parser == 'grid':
            grid(ns.model, ns.test_only, ns.output, ns.verbose)
        elif ns.parser == 'serve':
            serve()
        else:
            parser.print_help()
    except ConfigNotFound as e:
        print(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
