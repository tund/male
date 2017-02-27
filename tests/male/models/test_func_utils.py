from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np


def test_logsumeone():
    from male.utils.func_utils import logsumone
    assert (np.abs(logsumone(10.0) - 10.000045398899218) < 1e-8)
    print(logsumone(1000000.0))
    x = np.random.rand(2, 3)
    print(logsumone(x))


if __name__ == '__main__':
    pytest.main([__file__])
