from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from ..distributions import Distribution


class Uniform(Distribution):
    def __init__(self, low=(-1.0, -0.5), high=(1.0, 1.5), **kwargs):
        super(Uniform, self).__init__(**kwargs)
        self.low = low if isinstance(low, tuple) else (low,)
        self.high = high if isinstance(high, tuple) else (high,)
        self.dim = len(self.low)
        assert len(self.low) == len(self.high)

    def sample(self, num_samples, **kwargs):
        samples = np.zeros(shape=[num_samples, self.dim])
        for i in range(self.dim):
            unisampler = Uniform1D(low=self.low[i], high=self.high[i],
                                   random_state=self.random_engine)
            samples[:, i] = unisampler.sample(num_samples=num_samples)
        return samples

    def stratified_sample(self, num_samples):
        samples = np.zeros(shape=[num_samples, self.dim])
        for i in range(self.dim):
            unisampler = Uniform1D(low=self.low[i], high=self.high[i],
                                   random_state=self.random_engine)
            samples[:, i] = unisampler.stratified_sample(num_samples=num_samples)
        return samples

    def stratified_sample_v1(self, num_samples):
        samples = np.zeros(shape=[num_samples, self.dim])
        for i in range(self.dim):
            unisampler = Uniform1D(low=self.low[i], high=self.high[i],
                                   random_state=self.random_engine)
            samples[:, i] = unisampler.stratified_sample_v1(num_samples=num_samples)
        return samples

    def grid_sample(self, num_samples, bins=10):
        samples = np.zeros(shape=[num_samples, self.dim])
        for i in range(self.dim):
            unisampler = Uniform1D(low=self.low[i], high=self.high[i],
                                   random_state=self.random_engine)
            samples[:, i] = unisampler.grid_sample(num_samples=num_samples, bins=bins)
        return samples


class Uniform1D(Distribution):
    def __init__(self, low=-1.0, high=1.0, **kwargs):
        super(Uniform1D, self).__init__(**kwargs)
        self.low = low
        self.high = high

    def sample(self, num_samples, **kwargs):
        return self.random_engine.uniform(self.low, self.high, num_samples)

    def stratified_sample(self, num_samples):
        return np.linspace(self.low, self.high, num_samples) \
               + self.random_engine.random(num_samples) * 0.01

    def stratified_sample_v1(self, num_samples):
        s = np.linspace(self.low, self.high, num_samples)
        width = (self.high - self.low) / (
                num_samples - 1) if num_samples > 1 else self.high - self.low
        s[:-1] += self.random_engine.random(num_samples - 1) * width
        s[-1] -= self.random_engine.random(1) * width
        return s

    def grid_sample(self, num_samples, bins=10):
        grid = np.linspace(self.low, self.high, bins)
        idx = self.random_engine.randint(0, len(grid), size=num_samples)
        return grid[idx]
