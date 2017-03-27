from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .. import Optimizer


class Newton(Optimizer):
    """
    Newton's method
    """

    def __init__(self, tolerance=1e-7, max_loop=10, **kwargs):
        super(Newton, self).__init__(**kwargs)
        self.tolerance = tolerance
        self.max_loop = max_loop
        self.hess_func = None

    def init_params(self, hess_func=None, **kwargs):
        super(Newton, self).init_params(**kwargs)
        self.hess_func = hess_func

    def update_params(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, x=None, *args, **kwargs):
        obj_lst = []
        for l in range(self.max_loop):
            grad = self.grad_func(x, *args)
            hess = self.hess_func(x, *args)
            hess_inv = np.linalg.inv(hess)
            d = -np.dot(hess_inv, grad.reshape((len(x), 1)))
            d = d.reshape(len(x))
            t = 1
            obj_x = self.obj_func(x, *args)
            while self.obj_func(x + t * d, *args) > obj_x + 0.5 * t * np.dot(grad, d):
                t = self.learning_rate * t
            x += t * d
            obj_lst.append(self.obj_func(x, *args))
            if (l > 1) and np.abs((obj_lst[l] - obj_lst[l - 1]) / obj_lst[l]) < self.tolerance:
                break
        return x, obj_lst[-1]
