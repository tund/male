from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from os.path import expanduser

HOME = expanduser("~")

_EPSILON = 1e-8
_RANDOM_SEED = 6789
_DATA_DIR = os.path.join(HOME, ".male/datasets")
_MODEL_DIR = os.path.join(HOME, ".male/models")
_REMOTE_DATA_DIR = "http://prada-research.net/demo/datasets"
_REMOTE_MODEL_DIR = "http://prada-research.net/demo/models"


def epsilon():
    '''Returns the value of the fuzz
    factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> male.epsilon()
        1e-08
    ```
    '''
    return _EPSILON


def set_epsilon(e):
    '''Sets the value of the fuzz
    factor used in numeric expressions.

    # Arguments
        e: float. New value of epsilon.

    # Example
    ```python
        >>> from male import common as M
        >>> M.epsilon()
        1e-08
        >>> M.set_epsilon(1e-05)
        >>> M.epsilon()
        1e-05
    ```
    '''
    global _EPSILON
    _EPSILON = e


def random_seed():
    return _RANDOM_SEED


def set_random_seed(s):
    global _RANDOM_SEED
    _RANDOM_SEED = s


def data_dir():
    return _DATA_DIR


def set_data_dir(d):
    global _DATA_DIR
    _DATA_DIR = d


def model_dir():
    return _MODEL_DIR


def set_model_dir(d):
    global _MODEL_DIR
    _MODEL_DIR = d


def remote_data_dir():
    return _REMOTE_DATA_DIR


def set_remote_data_dir(d):
    global _REMOTE_DATA_DIR
    _REMOTE_DATA_DIR = d


def remote_model_dir():
    return _REMOTE_MODEL_DIR


def set_remote_model_dir(d):
    global _REMOTE_MODEL_DIR
    _REMOTE_MODEL_DIR = d
