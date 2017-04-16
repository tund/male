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
        return samples

    def logpdf(self, samples):
        p = np.zeros([samples.shape[0], len(self.pi)])
        for i in range(len(self.pi)):
            p[:, [i]] = self.pi[i] * norm.pdf(samples, loc=self.mu[i], scale=self.sigma[i])
        return np.mean(np.log(np.sum(p, axis=1)))


class GMM(object):
    """Gaussian Mixture Model - generic model including Multivariate GMM
    """

    def __init__(self, mix_coeffs=(0.5, 0.5),
                 mean=((0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
                 cov=((0.1, 0.14, 0.2),
                      (0.5, 0.2, 0.3))):
        self.mix_coeffs = mix_coeffs
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        assert len(mix_coeffs) == self.mean.shape[0]
        assert self.cov.shape[0] == self.mean.shape[0]
        assert self.cov.shape[1] == self.mean.shape[1]
        self.dim = self.mean.shape[1]

    def sample(self, num_samples):
        z = np.random.multinomial(num_samples, self.mix_coeffs)
        samples = np.zeros(shape=[num_samples, self.dim])
        i_start = 0
        for i in range(len(self.mix_coeffs)):
            i_end = i_start + z[i]
            samples[i_start:i_end, :] = np.random.multivariate_normal(
                mean=self.mean[i, :],
                cov=np.diag(self.cov[i, :]),
                size=z[i])
            i_start = i_end
        return samples

    def logpdf(self, samples):
        raise NotImplementedError
