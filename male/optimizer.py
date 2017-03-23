from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import abc


class Optimizer(object):
    """Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []
        self.grad

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_grad(self, loss, params):
        grads = K.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        """Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).

        # Raises
            ValueError: in case of incompatible weight shapes.
        """
        params = self.weights
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise ValueError('Optimizer weight shape ' +
                                 str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        K.batch_set_value(weight_value_tuples)

    def get_weights(self):
        """Returns the current value of the weights of the optimizer.

        # Returns
            A list of numpy arrays.
        """
        return K.batch_get_value(self.weights)

    def update_params(self, obj_func, params, grad_func=None, **kwargs):
        raise NotImplementedError

    def solve(self, obj_func, params, grad_func=None, callbacks=None, **kwargs):
        raise NotImplementedError

    def get_config(self):
        config = {}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
