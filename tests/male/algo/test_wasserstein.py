from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np
from male.algo import emd


def test_small():
    np.random.seed(6789)
    # m = 3
    # n = 4
    a = [0.2, 0.3, 0.5]
    b = [0.1, 0.15, 0.3, 0.45]
    c = np.random.rand(3, 4)
    result = emd(a, b, c)
    print("cost = {}".format(result))
    assert abs(result - 0.231071125718) < 1e-8


def test_big():
    np.random.seed(6789)
    m = 95
    n = 100
    a = np.random.rand(m)
    a /= np.sum(a)
    b = np.random.rand(n)
    b /= np.sum(b)
    c = 100 * np.random.rand(m, n)
    result = emd(a, b, c)
    print("cost = {}".format(result))
    assert abs(result - 3.10073169288) < 1e-8


if __name__ == '__main__':
    pytest.main([__file__])
    # test_small()
    # test_big()
