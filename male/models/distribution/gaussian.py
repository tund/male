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
