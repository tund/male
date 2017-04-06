from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics

from male.datasets import demo
from male.models.kernel import KSGD
from male.callbacks import Display


def test_sgd_visualization_2d():
    (x_train, y_train), (_, _) = demo.load_synthetic_2d()

    display = Display(freq=10,
                      dpi='auto',
                      monitor=[{'metrics': ['predict'],
                                'title': "Learning losses",
                                'xlabel': "X1",
                                'ylabel': "X2",
                                'grid_size': 100,
                                'marker_size': 10,
                                'left': None,
                                'right': None,
                                'top': None,
                                'bottom': None
                                }]
                      )

    learner = KSGD(lbd=0.0001,
                   eps=0.001,
                   gamma=30,
                   kernel='gaussian',
                   loss='hinge',
                   batch_size=1,
                   callbacks=[display],
                   avg_weight=False)

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))


def test_sgd_svmguide1_bin():
    (x_train, y_train), (x_test, y_test) = demo.load_svmguide1()

    learner = KSGD(lbd=0.1,
                   eps=0.01,
                   gamma=2,
                   kernel='gaussian',
                   loss='hinge',
                   batch_size=10,
                   avg_weight=False)

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_sgd_visualization_2d()
    # test_sgd_svmguide1_bin()
