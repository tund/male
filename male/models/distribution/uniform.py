from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


class Uniform1D(object):
    def __init__(self, low=-1.0, high=1.0):
        self.low = low
        self.high = high

    def sample(self, num_samples):
        return np.random.uniform(self.low, self.high, num_samples)

    def stratified_sample(self, num_samples):
        return np.linspace(self.low, self.high, num_samples) \
               + np.random.random(num_samples) * 0.01

    def stratified_sample_v1(self, num_samples):
        s = np.linspace(self.low, self.high, num_samples)
        width = (self.high - self.low) / (
            num_samples - 1) if num_samples > 1 else self.high - self.low
        s[:-1] += np.random.random(num_samples - 1) * width
        s[-1] -= np.random.random(1) * width
        return s
