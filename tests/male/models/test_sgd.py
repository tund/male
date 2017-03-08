from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedShuffleSplit

from male.models.kernel import KSGD

data_dir = 'C:/Data/'


def test_sgd_svmguide1_bin():
    data_name = 'svmguide1'
    n_features = 4
    train_file_name = data_dir + data_name + '.txt'
    test_file_name = data_dir + data_name + '_t.txt'

    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

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
    # pytest.main([__file__])
    test_sgd_svmguide1_bin()
