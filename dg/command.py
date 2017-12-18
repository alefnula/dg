__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = '18 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import inspect
import weakref


class Command(object):
    __instances = set()

    def __init__(self, target_fn):
        self.target_fn = target_fn
        self.name = target_fn.__name__
        self.help = target_fn.__doc__.splitlines()[0]
        self.signature = inspect.signature(target_fn)
        self.arguments = getattr(target_fn, 'arguments', [])
        self.__instances.add(weakref.ref(self))

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return f'{self.name}{self.signature}'

    def __call__(self, *args, **kwargs):
        return self.target_fn(*args, **kwargs)

    def run(self, namespace):
        data = {}
        for param in self.signature.parameters:
            data[param] = getattr(namespace, param)
        return self.target_fn(**data)

    def create_subparser(self, parser):
        subparser = parser.add_parser(self.name, help=self.help)
        for args, kwargs in reversed(self.arguments):
            subparser.add_argument(*args, **kwargs)
        return subparser

    @classmethod
    def get_instances(cls):
        dead = set()
        alive = {}
        for ref in cls.__instances:
            obj = ref()
            if obj is not None:
                alive[obj.name] = obj
            else:
                dead.add(ref)
        cls.__instances -= dead
        return alive


def command(func):
    return Command(func)


def argument(*args, **kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('Argument options must be specified')

    def decorator(func):
        if not hasattr(func, 'arguments'):
            func.arguments = []
        func.arguments.append((args, kwargs))
        return func
    return decorator
