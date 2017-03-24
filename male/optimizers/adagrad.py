from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .. import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, epsilon=1e-8, decay=0., **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.decay = decay

    def init_params(self, **kwargs):
        super(Adagrad, self).init_params(**kwargs)
        self.iteration = 0
        self.initial_decay = self.decay
        self.accumulators = [np.zeros(p.shape) for p in self.params]

    def update_params(self, *args, **kwargs):
        grads = self.grad_func(*args)

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iteration))
            self.iteration += 1

        for p, g, a in zip(self.params, grads, self.accumulators):
            a[:] = a + g * g  # update accumulator
            p[:] = p - lr * g / (np.sqrt(a) + self.epsilon)
