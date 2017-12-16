__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c) 2017 Viktor Kerkez'

import os
from tea import shell
from datetime import datetime

GIT_IGNORE = '''\
*.py[cod]
.idea
.vscode
.ipynb_checkpoints
data/*
models/*
'''

SIGNATURE = '''\
__author__ = '{author} <{email}>'
__date__ = '{now:%d %B %Y}'
__copyright__ = 'Copyright (c) {now:%Y} {author}'
'''

REQUIREMENTS = '''\
dg
'''

CONFIG = '''\
models:
'''


def create(project=None, author=None, email=None):
    """Creates the project skeleton.

    Args:
        project (str): Project name or path to the project directory
        author (str): Author's full name
        email (str): Author's email address
    """
    if project is None:
        project_name = input('Project name:')
        project_dir = os.path.abspath(os.path.join(os.getcwd(), project_name))
    else:
        project_name = os.path.basename(project)
        if project_name != project:
            project_dir = os.path.abspath(project)
        else:
            project_dir = os.path.abspath(
                os.path.join(os.getcwd(), project_name)
            )

    if author is None:
        author = input('Author:')

    if email is None:
        email = input('Email:')

    # Create the project skeleton
    print(f'Creating project: {project_dir}')
    shell.mkdir(project_dir)
    shell.touch(os.path.join(project_dir, '.gitignore'),
                content=GIT_IGNORE)
    shell.touch(os.path.join(project_dir, 'requirements.txt'),
                content=REQUIREMENTS)
    # Create configuration
    config_dir = os.path.join(project_dir, 'config')
    shell.mkdir(config_dir)
    shell.touch(os.path.join(config_dir, f'{project_name}.yaml'),
                content=CONFIG)
    # Create data directory
    data_dir = os.path.join(project_dir, 'data')
    shell.mkdir(data_dir)
    shell.touch(os.path.join(data_dir, '.keep'))
    # Create models directory
    models_dir = os.path.join(project_dir, 'models')
    shell.mkdir(models_dir)
    shell.touch(os.path.join(models_dir, '.keep'))
    shell.mkdir(os.path.join(project_dir, project_name))
    # Create the app
    signature = SIGNATURE.format(author=author, email=email,
                                 now=datetime.now())
    app_dir = os.path.join(project_dir, project_name)
    shell.mkdir(app_dir)
    shell.touch(os.path.join(app_dir, '__init__.py'),
                content=signature)
    # Create models subdirectory
    shell.mkdir(os.path.join(app_dir, 'models'))
    shell.touch(os.path.join(app_dir, 'models', '__init__.py'),
                content=signature)
    # Create commands subdirectory
    shell.mkdir(os.path.join(app_dir, 'commands'))
    shell.touch(os.path.join(app_dir, 'commands', '__init__.py'),
                content=signature)