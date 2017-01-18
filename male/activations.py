from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

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
