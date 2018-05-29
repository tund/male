from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ..distributions import Distribution


class Multinomial(Distribution):

    def __init__(self, probs, **kwargs):
        super(Multinomial, self).__init__(**kwargs)
        self.pi = probs / np.sum(probs)
        self._setup_alias()

    def _setup_alias(self):
        K = len(self.pi)
        self.q = np.zeros(K)
        self.J = np.zeros(K, dtype=np.int)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(self.pi):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = self.q[large] - (1.0 - self.q[small])

            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def sample(self, num_samples, **kwargs):
        K = len(self.J)

        # Draw from the overall uniform mixture.
        kk = np.floor(self.random_engine.rand(num_samples) * K).astype(np.int)

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        idx = self.random_engine.rand(num_samples) >= self.q[kk]
        kk[idx] = self.J[kk[idx]]
        # if self.random_engine.rand(num_samples) < self.q[kk]:
        #     return kk
        # else:
        #     return self.J[kk]
        return kk
