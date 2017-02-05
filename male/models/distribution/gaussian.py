from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy.stats import norm


class Gaussian1D(object):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        samples = np.random.normal(self.mu, self.sigma, num_samples)
        return samples

    def logpdf(self, samples):
        return np.mean(np.log(norm.pdf(samples, loc=self.mu, scale=self.sigma)))


class InverseGaussian1D(object):
    def __init__(self, mu=0.0, sigma=1.0, low=-6, high=6):
        self.mu = mu
        self.sigma = sigma
        self.low = low
        self.high = high

    def sample(self, num_samples, num_bins=10):
        s = np.random.normal(self.mu, self.sigma, num_samples)
        bins = np.linspace(self.low, self.high, num_bins)
        pd, _ = np.histogram(s, bins=bins, density=True)
        pd /= np.sum(pd)
        pd = 1 - pd
        pd /= np.sum(pd)
        z = np.random.multinomial(num_samples, pd)
        samples = []
        for i in range(len(z)):
            samples.extend(bins[i] + np.random.random(z[i]) * 0.01)
        return samples
