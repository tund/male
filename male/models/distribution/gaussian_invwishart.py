from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as stats


class GaussianInvWishart(object):
    """Gaussian Inverse Wishart distribution
    Prioritize speed
    """
    def __init__(self, dim, loc, lbd, degree, scale_mat):
        """
        Complete Initialize
        :param dim: dimension of data
        :param loc: location of data (d,) or scalar
        :param lbd: scalar
        :param degree: degree of freedom, scalar > d - 1
        :param scale_mat: (d,d)
        """
        self.dim = dim
        if degree <= dim - 1:
            raise ValueError("degree must be greater than dim - 1")
        if np.isscalar(loc):
            loc *= np.ones(dim)

        self.loc0 = loc.copy()
        self.lbd0 = lbd
        self.degree0 = degree
        self.scale_mat = scale_mat.copy()

        self.num_samples = 0
        self.x_sum = np.zeros(dim)
        self.loc = self.loc0
        self.lbd = self.lbd0
        self.degree = self.degree0
        self.scale_mat = self.scale_mat
        self.x_lst = []
        self.invalid = False

    def sample(self):
        """
        Sample mean & cov from (mu, lbd, nu, psi_m)
        :return: mean, cov
        """
        if self.invalid:
            self.update()
        cov = stats.invwishart.rvs(df=self.degree, scale=self.scale_mat)
        mean = stats.multivariate_normal.rvs(mean=self.loc, cov=cov / self.lbd)
        return mean, cov

    def update(self):
        """
        Up to date parameters after add or delete item
        :return:
        """
        if self.num_samples > 0:
            x_bar = self.x_sum / self.num_samples
            dx_bar = self.x_lst - x_bar  # array (n,d)
            C = np.dot(dx_bar.T, dx_bar)  # array (d,d)
        else:
            x_bar = np.zeros(self.dim)
            C = np.zeros((self.dim, self.dim))
        x_bar_mu = (x_bar - self.loc0).reshape((self.dim, 1))
        self.loc = (self.lbd0 * self.loc0 + self.x_sum) / self.lbd
        self.scale_mat = self.scale_mat + C + (self.lbd0 * self.num_samples) * np.dot(x_bar_mu, x_bar_mu.T) / self.lbd

    def add_item(self, x):
        """
        Add one item to component
        :param x: (d,)
        :return: index of item in the component
        """
        self.num_samples += 1
        self.lbd += 1
        self.degree += 1
        self.x_sum += x
        self.x_lst.append(x)
        self.invalid = True
        return self.num_samples - 1  # zero-based index

    def del_item(self, index):
        """
        Delete one item from component
        :param index: (d,)
        :return:
        """
        self.num_samples -= 1
        self.lbd -= 1
        self.degree -= 1
        self.x_sum -= self.x_lst[index]
        del self.x_lst[index]
        self.invalid = True
