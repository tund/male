from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male.configs import random_seed


def test_tuid():
    from male.utils.generic_utils import tuid
    print(tuid())


def test_logsumeone():
    np.random.seed(random_seed())
    from male.utils.func_utils import logsumone
    assert (np.abs(logsumone(10.0) - 10.000045398899218) < 1e-6)
    x = np.random.rand(2, 3)
    r = np.array([[1.30170729, 1.0712702, 1.00492985],
                  [1.14584669, 1.10508236, 0.94158162]])
    assert np.all(np.abs(logsumone(x) - r) < 1e-2)


def test_label_encoder_dict():
    from male.utils.label import LabelEncoderDict
    y = np.array([-1, 1, 3, 3, 1, 20])
    encoder = LabelEncoderDict()
    y_encoded = encoder.fit_transform(y)
    assert((y_encoded == np.array([0, 1, 2, 2, 1, 3])).all())

    y_encoded = encoder.transform(np.array([3, 20]))
    assert ((y_encoded == np.array([2, 3])).all())
    y_decoded = encoder.inverse_transform(np.array([0, 1, 1, 3, 2]))
    assert ((y_decoded == np.array([-1, 1, 1, 20, 3])).all())

if __name__ == '__main__':
    pytest.main([__file__])
    # test_tuid()
    # test_logsumeone()
    # test_predict_visualization(show=True, block_figure_on_end=True)
    # test_label_encoder_dict()
