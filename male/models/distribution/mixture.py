from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy.stats import norm


class GMM1D(object):
    """Gaussian Mixture Model 1D (Univariate GMM)
    """

    def __init__(self, pi=(0.5, 0.5), mu=(0.0, 4.0), sigma=(1.0, 0.5)):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        z = np.random.multinomial(num_samples, self.pi)
        samples = np.array([])
        for i in range(len(self.pi)):
            samples = np.concatenate([samples, np.random.normal(self.mu[i], self.sigma[i], z[i])])
        # samples.sort()
        return samples

    def logpdf(self, samples):
        p = np.zeros([samples.shape[0], len(self.pi)])
        for i in range(len(self.pi)):
            p[:, [i]] = self.pi[i] * norm.pdf(samples, loc=self.mu[i], scale=self.sigma[i])
        return np.mean(np.log(np.sum(p, axis=1)))
