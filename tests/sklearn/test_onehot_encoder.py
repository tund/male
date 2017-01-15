from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def test_onehot_encoder():
    y = np.random.randint(0, 10, 1000)
    e = OneHotEncoder()
    t = e.fit_transform(y.reshape(-1, 1)).toarray()
    assert np.all(np.argmax(t, axis=1) == y)


if __name__ == '__main__':
    pytest.main(__file__)
