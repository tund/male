from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from sklearn import metrics

from male.models.kernel.bank import BaNK
from male.utils.disp_utils import visualize_classification_prediction
from male.datasets import demo


def test_bank_2d(show=False, block_figure_on_end=False):
    print("========== Test BaNK on 2D data ==========")

    (x_train, y_train), (_, _) = demo.load_synthetic_2d()

    # Being careful at this point
    y_train[y_train == -1] = 0

    learner = BaNK(
        gamma=20,
        rf_dim=400,
        inner_regularization=0.125,
        alpha=1.0,
        kappa=1.0,
        inner_max_loop=1,
        max_outer_loop=2,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    print(np.unique(y_train))
    print(np.unique(y_train_pred))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    visualize_classification_prediction(learner, x_train, y_train, grid_size=50, show=show,
                                        epoch=0, marker_size=20, block_on_end=block_figure_on_end)


def test_bank_svmguide1():
    print("========== Test BaNK on svmguide1 dataset ==========")

    data_name = 'svmguide1'

    (x_train, y_train), (x_test, y_test) = demo.load_svmguide1()

    learner = BaNK(
        gamma=1.0 / 1.25727,
        rf_dim=384,
        inner_regularization=0.125,
        outer_regularization=0.125,
        alpha=10.0,
        kappa=0.1,
        inner_max_loop=1,
        max_outer_loop=5,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_bank_2d(show=True, block_figure_on_end=True)
    # test_bank_svmguide1()
