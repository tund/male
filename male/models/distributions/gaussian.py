from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from ...configs import epsilon
from ..distributions import Distribution


class Gaussian1D(Distribution):

    def __init__(self, mu=0.0, sigma=1.0, **kwargs):
        super(Gaussian1D, self).__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples, **kwargs):
        samples = self.random_engine.normal(self.mu, self.sigma, num_samples)
        return samples

    def logpdf(self, samples):
        return np.mean(np.log(norm.pdf(samples, loc=self.mu, scale=self.sigma) + epsilon()))


class InverseGaussian1D(Distribution):
    def __init__(self, mu=0.0, sigma=1.0, low=-6, high=6, **kwargs):
        super(InverseGaussian1D, self).__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high

    def sample(self, num_samples, num_bins=10, **kwargs):
        s = self.random_engine.normal(self.mu, self.sigma, num_samples)
        bins = np.linspace(self.low, self.high, num_bins)
        pd, _ = np.histogram(s, bins=bins, density=True)
        pd /= np.sum(pd)
        pd = 1 - pd
        pd /= np.sum(pd)
        z = self.random_engine.multinomial(num_samples, pd)
        samples = []
        for i in range(len(z)):
            samples.extend(bins[i] + self.random_engine.random(z[i]) * 0.01)
        return samples


class Gaussian(Distribution):
    """Multivariate Gaussian Distribution
    """

    def __init__(self, mu=(0.0, 0.0), sigma=(1.0, 1.0), **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma
        self.cov = np.diag(sigma)

    def sample(self, num_samples, **kwargs):
        return self.random_engine.multivariate_normal(mean=self.mu, cov=self.cov, size=num_samples)

    def logpdf(self, samples):
        return np.mean(np.log(multivariate_normal.pdf(samples, mean=self.mu, cov=self.sigma)
                              + epsilon()))
