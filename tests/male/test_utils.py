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


def test_predict_visualization(show=False, block_figure_on_end=False):
    from male.datasets import demo
    from male.models.kernel import KSGD
    from male.utils.disp_utils import visualize_classification_prediction

    (x_train, y_train), (_, _) = demo.load_synthetic_2d()

    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    learner = KSGD(lbd=0.001,
                   eps=0.001,
                   gamma=20,
                   kernel='gaussian',
                   loss='hinge',
                   batch_size=1,
                   avg_weight=False)

    learner.fit(x_train, y_train)

    visualize_classification_prediction(learner, x_train, y_train, show=show,
                                        block_on_end=block_figure_on_end,
                                        epoch=learner.epoch, grid_size=20)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_tuid()
    # test_logsumeone()
    # test_predict_visualization(show=True, block_figure_on_end=True)
