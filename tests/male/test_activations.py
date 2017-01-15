import pytest
import numpy as np
from numpy.testing import assert_allclose

from male import activations


def test_softmax():
    x = np.random.rand(10, 20)
    assert_allclose(activations.softmax(x).sum(axis=1), np.ones(10), rtol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])
