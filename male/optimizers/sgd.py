from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

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

    def __init__(self, learning_rate=0.01, momentum=0.,
                 decay=0., nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.iterations = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.initial_decay = decay
        self.nesterov = nesterov

    def update_params(self, obj_func, params, grad_func=None, *args, **kwargs):
        grads = grad_func(*args, **kwargs)
        self.updates = []

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.iterations += 1

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
