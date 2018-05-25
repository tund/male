from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from ...configs import epsilon
from ..distributions import Distribution


class GMM1D(Distribution):
    """Gaussian Mixture Model 1D (Univariate GMM)
    """

    def __init__(self, pi=(0.5, 0.5), mu=(0.0, 4.0), sigma=(1.0, 0.5), **kwargs):
        super(GMM1D, self).__init__(**kwargs)
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples, **kwargs):
        z = self.random_engine.multinomial(num_samples, self.pi)
        samples = np.array([])
        for i in range(len(self.pi)):
            samples = np.concatenate([samples,
                                      self.random_engine.normal(self.mu[i], self.sigma[i], z[i])])
        return samples

    def logpdf(self, samples):
        p = np.zeros([samples.shape[0], len(self.pi)])
        for i in range(len(self.pi)):
            p[:, [i]] = self.pi[i] * norm.pdf(samples, loc=self.mu[i], scale=self.sigma[i])
        return np.mean(np.log(np.sum(p, axis=1) + epsilon()))


class GMM(Distribution):
    """Gaussian Mixture Model - generic model including Multivariate GMM
    """

    def __init__(self, mix_coeffs=(0.5, 0.5),
                 mean=((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
                 cov=((0.1, 0.14, 0.2),
                      (0.5, 0.2, 0.3)), **kwargs):
        super(GMM, self).__init__(**kwargs)
        self.mix_coeffs = mix_coeffs
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        assert len(mix_coeffs) == self.mean.shape[0]
        assert self.cov.shape[0] == self.mean.shape[0]
        assert self.cov.shape[1] == self.mean.shape[1]
        self.dim = self.mean.shape[1]

    def sample(self, num_samples, **kwargs):
        z = self.random_engine.multinomial(num_samples, self.mix_coeffs)
        samples = np.zeros(shape=[num_samples, self.dim])
        i_start = 0
        for i in range(len(self.mix_coeffs)):
            i_end = i_start + z[i]
            samples[i_start:i_end, :] = self.random_engine.multivariate_normal(
                mean=self.mean[i, :],
                cov=np.diag(self.cov[i, :]),
                size=z[i])
            i_start = i_end
        return samples

    def logpdf(self, samples):
        p = np.zeros([samples.shape[0], len(self.mix_coeffs)])
        for i in range(len(self.mix_coeffs)):
            p[:, i] = self.mix_coeffs[i] * multivariate_normal.pdf(
                samples, mean=self.mean[i], cov=self.cov[i])
        return np.mean(np.log(np.sum(p, axis=1) + epsilon()))
