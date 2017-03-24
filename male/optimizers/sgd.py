from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .. import Optimizer


class SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, momentum=0., decay=0., nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov

    def init_params(self, **kwargs):
        super(SGD, self).init_params(**kwargs)
        self.iter_ = 0
        self.initial_decay_ = self.decay
        self.moments_ = [np.zeros(p.shape) for p in self.params_]

    def update_params(self, *args, **kwargs):
        grads = self.grad_func_(*args)

        lr = self.learning_rate
        if self.initial_decay_ > 0:
            lr *= (1. / (1. + self.decay * self.iter_))
            self.iter_ += 1

        # momentum
        for p, g, m in zip(self.params_, grads, self.moments_):
            m[:] = self.momentum * m - lr * g  # velocity
            if self.nesterov:
                new_p = p + self.momentum * m - lr * g
            else:
                new_p = p + m
            p[:] = new_p
