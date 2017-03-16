from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as stats


class GaussianInvWishart(object):
    """Gaussian Inverse Wishart distribution
    Prioritize speed
    """
    def __init__(self, d, mu, lbd, nu, psi_m):
        """
        Complete Initialize
        :param d: dimension
        :param mu: (d,) or scalar
        :param lbd: scalar
        :param nu: scalar > d - 1
        :param psi_m: (d,d)
        """
        self.d = d
        if nu <= d - 1:
            raise ValueError("nu must be greater than d - 1")
        if np.isscalar(mu):
            mu *= np.ones(d)

        self.mu0 = mu.copy()
        self.lbd0 = lbd
        self.nu0 = nu
        self.psi0_m = psi_m.copy()

        self.n = 0
        self.x_sum = np.zeros(d)
        self.mu = self.mu0
        self.lbd = self.lbd0
        self.nu = self.nu0
        self.psi_m = self.psi0_m
        self.x_lst = []
        self.invalid = False

    def sample(self):
        """
        Sample mean & cov from (mu, lbd, nu, psi_m)
        :return: mean, cov
        """
        if self.invalid:
            self.update()
        cov = stats.invwishart.rvs(df=self.nu, scale=self.psi_m)
        mean = stats.multivariate_normal.rvs(mean=self.mu, cov=cov)
        return mean, cov

    def update(self):
        """
        Up to date parameters after add or delete item
        :return:
        """
        if self.n > 0:
            x_bar = self.x_sum / self.n
            dx_bar = self.x_lst - x_bar  # array (n,d)
            C = np.dot(dx_bar.T, dx_bar)  # array (d,d)
        else:
            x_bar = np.zeros(self.d)
            C = np.zeros(self.d, self.d)
        x_bar_mu = (x_bar - self.mu0).reshape((self.d,1))
        self.mu = (self.lbd0 * self.mu0 + self.x_sum) / self.lbd
        self.psi_m = self.psi0_m + C + (self.lbd0 * self.n) * np.dot(x_bar_mu, x_bar_mu.T) / self.lbd

    def add_item(self, x):
        """
        Add one item to component
        :param x: (d,)
        :return:
        """
        self.n += 1
        self.lbd += 1
        self.nu += 1
        self.x_sum += x
        self.x_lst.append(x)
        self.invalid = True

    def del_item(self, index):
        """
        Delete one item from component
        :param index: (d,)
        :return:
        """
        self.n -= 1
        self.lbd -= 1
        self.nu -= 1
        self.x_sum -= self.x_lst[index]
        del self.x_lst[index]
        self.invalid = True

