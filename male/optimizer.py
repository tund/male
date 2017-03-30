from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc
from .utils.generic_utils import get_from_module


class Optimizer(object):
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate=0.01, **kwargs):
        self.__dict__.update(kwargs)
        self.learning_rate = learning_rate

    def init_params(self, obj_func=None, grad_func=None, params=[], **kwargs):
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.params = params

    def update_params(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, callbacks=None, *args, **kwargs):
        raise NotImplementedError


def get(identifier, kwargs=None):
    from . import optimizers

    sgd = optimizers.SGD
    adam = optimizers.Adam
    adamax = optimizers.Adamax
    nadam = optimizers.Nadam
    adagrad = optimizers.Adagrad
    rmsprop = optimizers.RMSProp
    adadelta = optimizers.Adadelta

    if identifier not in locals().keys():
        return identifier
    else:
        # Instantiate a Male optimizer
        return get_from_module(identifier, locals(), 'optimizer',
                               instantiate=True, kwargs=kwargs)
