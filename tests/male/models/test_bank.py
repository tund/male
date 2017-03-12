from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from sklearn import metrics

from sklearn.datasets import load_svmlight_file

from male.models.kernel.bank import BANK
from male.utils.disp_utils import visualize_classification_prediction

data_dir = 'C:/Data/'


def test_bank_2d():
    sub_folder = '2d/'
    data_name = 'train.bin'
    n_features = 2
    train_file_name = data_dir + sub_folder + data_name + '.txt'
    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_train = x_train.toarray()
    # CARE
    y_train[y_train == -1] = 0

    learner = BANK(
        gamma=20,
        dim_rf=400,
        lbd=0.001,
        alpha=1.0,
        kappa=0.1,
        inner_epoch=1,
        max_loop=20,
        batch_size=100
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    visualize_classification_prediction(learner, x_train, y_train, epoch=0, marker_size=20)


def test_bank_svmguide1():
    sub_folder = ''
    data_name = 'svmguide1'
    n_features = 4
    train_file_name = data_dir + sub_folder + data_name + '.txt'
    test_file_name = data_dir + sub_folder + data_name + '_t.txt'

    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    learner = BANK(
        gamma=20,
        dim_rf=400,
        lbd=0.001,
        alpha=1.0,
        kappa=0.1,
        inner_epoch=1,
        max_loop=20,
        batch_size=100
    )

    learner.fit(x_train, y_train)

    y_test_pred = learner.predict(x_test)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    # pytest.main([__file__])
    # test_bank_2d()
    test_bank_svmguide1()
