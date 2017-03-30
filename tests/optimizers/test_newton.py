from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from male.optimizers.newton import Newton

import numpy as np


# f1 = x^2+y^2-3x+6y
# min = -11.25 at x = 1.5, y = -3
def obj_func_f1(x, (a, b)):
    return x[0]**2 + (x[1]**2) + a*x[0] + b*x[1]


def grad_func_f1(x, (a, b)):
    return np.array([2*x[0] + a, 2*x[1] + b])


def hess_func_f1(x, (a, b)):
    return np.diag(np.array([2, 2]))


def test_f1():
    opt = Newton(learning_rate=0.8, tolerance=1e-5, max_loop=10)
    opt.init_params(
        obj_func=obj_func_f1,
        grad_func=grad_func_f1,
        hess_func=hess_func_f1
    )
    x = np.array([10.0, 10.0])
    a = -3
    b = 6
    x, obj = opt.solve(x, (a, b))
    print(x, obj)

if __name__ == '__main__':
    test_f1()
