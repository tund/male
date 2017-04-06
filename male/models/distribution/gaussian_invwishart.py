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

    @staticmethod
    def sample_posterior(dim, data, lbd0, loc0, degree0, scale_mat0):
        """
        Sample from posterior distribution given observed data
        without requiring any immediate function call (e.g. add, del items)
        :param dim: dimension of data
        :param data: (num_samples,dim)
        :param lbd0: scalar
        :param loc0: location of data (dim,)
        :param degree0: degree of freedom, scalar > dim - 1
        :param scale_mat0: (dim,dim)
        :return: [mean_new, cov_new] of normal distribution
        """
        num_samples = data.shape[0]
        lbd0_loc0 = lbd0 * loc0
        if num_samples > 0:
            data_bar = np.mean(data, axis=0)  # (dim,)
            d_data_bar = data - data_bar  # (dim,num_samples)
            C = np.dot(d_data_bar.T, d_data_bar)  # (dim,dim)
        else:
            data_bar = np.zeros(dim)
            C = np.zeros((dim, dim))
        loc_new = (lbd0_loc0 + num_samples * data_bar) / (lbd0 + num_samples)
        degree_new = degree0 + num_samples
        kappa_new = lbd0 + num_samples
        data_bar_loc = (data_bar - loc0).reshape((dim, 1))
        scale_mat_new = \
            scale_mat0 + C + (lbd0 * num_samples / (lbd0 + num_samples)) * np.dot(data_bar_loc, data_bar_loc.T)
        cov_new = stats.invwishart.rvs(df=degree_new, scale=scale_mat_new)
        mean_new = stats.multivariate_normal.rvs(loc_new, cov_new / kappa_new)
        return [mean_new, cov_new]
