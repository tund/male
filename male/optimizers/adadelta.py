from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from . import RMSProp


class Adadelta(RMSProp):
    """Adadelta optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
            It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    """

    def __init__(self, learning_rate=1.0, rho=0.95, **kwargs):
        kwargs['learning_rate'] = learning_rate
        kwargs['rho'] = rho
        super(Adadelta, self).__init__(**kwargs)

    def init_params(self, **kwargs):
        super(Adadelta, self).init_params(**kwargs)
        self.delta_accumulators = [np.zeros(p.shape) for p in self.params]

    def update_params(self, *args, **kwargs):
        grads = self.grad_func(*args)

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iteration))
            self.iteration += 1

        for p, g, a, d_a in zip(self.params, grads, self.accumulators, self.delta_accumulators):
            # update accumulator
            a[:] = self.rho * a + (1. - self.rho) * g * g
            # use the new accumulator and the *old* delta_accumulator
            update = g * np.sqrt(d_a + self.epsilon) / np.sqrt(a + self.epsilon)
            p[:] = p - lr * update
            # update delta_accumulator
            d_a[:] = self.rho * d_a + (1 - self.rho) * np.square(update)
