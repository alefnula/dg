__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 16 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import os


def ensure_dir(path, directory=False):
    """Ensures that the directory of the file exists.

    Args:
        path (str): Path to the file for which we want to ensure the existence
            of a parent directory.
        directory (bool): If this is true the path is treated as the directory
            which existence should be ensured.
    Returns:
        str: Same path that we got as input but converted to absolute path
    """
    full_path = os.path.abspath(path)
    if directory:
        dirname = full_path
    else:
        dirname = os.path.dirname(full_path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return full_path


def print_and_save_df(df, output=None):
    """Print out the data frame to console and save it to file
    Args:
        df (pandas.DataFrame): DataFrame to be saved and printend
        output (str): Path oto the output CSV file
    """
    if output:
        output = ensure_dir(output)
        df.to_csv(output, index=False, float_format='%.4f')
    print(df.to_string(justify='right', float_format='%.4f', index=False))


def bar(n=79, verbose=True):
    """Prints a horizontal bar

    Args:
        n (int): Length of the bar
        verbose (bool): Condition for printing the bar:
    """
    if verbose:
        print('\n' + '-' * n + '\n')
