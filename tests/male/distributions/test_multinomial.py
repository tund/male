from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male.models.distributions import Multinomial


def test_sampling():
    K = 5
    N = 1000

    # Get a random probability vector.
    probs = np.random.dirichlet(np.ones(K), 1).ravel()

    # Construct the table.
    m = Multinomial(probs)

    # Generate variates.
    X = m.sample(N)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_sampling()
