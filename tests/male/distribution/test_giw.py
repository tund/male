from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np
import scipy.stats as stats

from male.models.distribution.gaussian_invwishart import GaussianInvWishart as GIW

# TODO:
# - Visualization how GIW change when add or delete observed data


def test_functionality_giw():
    giw = GIW(dim=2, loc=0, lbd=0.1, degree=3, scale_mat=2*np.eye(2))
    mean, cov = giw.sample()
    print('mean={}\ncov={}'.format(mean, cov))
    for i in range(5):
        x = stats.multivariate_normal.rvs(np.ones(2))
        giw.add_item(x)
    mean, cov = giw.sample()
    print('mean={};\ncov={}'.format(mean, cov))

    for i in range(2):
        x = stats.multivariate_normal.rvs(10*np.ones(2))
        giw.add_item(x)

    giw.del_item(5)
    giw.del_item(5)

    mean, cov = giw.sample()
    print('mean={};\ncov={}'.format(mean, cov))

if __name__ == '__main__':
    test_functionality_giw()
