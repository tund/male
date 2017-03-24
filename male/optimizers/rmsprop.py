from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from . import Adagrad


class RMSProp(Adagrad):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, learning_rate=0.001, rho=0.9, **kwargs):
        kwargs['learning_rate'] = learning_rate
        super(RMSProp, self).__init__(**kwargs)
        self.rho = rho

    def update_params(self, *args, **kwargs):
        grads = self.grad_func_(*args)

        lr = self.learning_rate
        if self.initial_decay_ > 0:
            lr *= (1. / (1. + self.decay * self.iter_))
            self.iter_ += 1

        for p, g, a in zip(self.params_, grads, self.accumulators_):
            # update accumulator
            a[:] = self.rho * a + (1. - self.rho) * g * g
            p[:] = p - lr * g / (np.sqrt(a) + self.epsilon)
