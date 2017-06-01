from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # no tensorflow module
    tf = None

from .utils.generic_utils import get_from_module

EPS = np.finfo(np.float32).eps


def sigmoid(x):
    """Compute sigmoid function: y = 1 / (1 + exp(-x))
    """
    max0 = np.maximum(x, 0)
    return np.exp(x - max0) / (np.exp(x - max0) + np.exp(-max0))


def softmax(x):
    xx = np.exp(x - np.max(x, 1, keepdims=True))
    return xx / (np.sum(xx, 1, keepdims=True) + EPS)


def logsumone(x):
    """Compute log(1 + exp(x))
    """
    max0 = np.maximum(x, 0)
    return np.log(np.exp(-max0) + np.exp(x - max0)) + max0


def tf_relu(x):
    return tf.nn.relu(x)


# <editor-fold desc="This is an implementation for general case for every alpha">
# def tf_lrelu(x, alpha=0.01):
#     if alpha != 0.:
#         negative_part = tf.nn.relu(-x)
#     x = tf.nn.relu(x)
#     if alpha != 0.:
#         x -= alpha * negative_part
#     return x
# </editor-fold>


# For alpha <= 1:
def tf_lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def tf_tanh(x):
    return tf.tanh(x)


def tf_sigmoid(x):
    return tf.sigmoid(x)


def linear(x):
    return x


def get(identifier):
    if identifier is None:
        return linear
    return get_from_module(identifier, globals(), 'activation function')
