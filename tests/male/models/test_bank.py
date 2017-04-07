from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np
import os

from sklearn import metrics
from sklearn.datasets import load_svmlight_file

from male.models.kernel.bank import BaNK
from male.utils.disp_utils import visualize_classification_prediction
from male.common import data_dir
from male.datasets import demo


def test_bank_2d():
    data_name = 'synthetic_2d_data'
    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')

    n_features = 2
    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_train = x_train.toarray()
    # CARE
    y_train[y_train == -1] = 0

    learner = BaNK(
        gamma=20,
        rf_dim=400,
        inner_regularization=0.125,
        alpha=1.0,
        kappa=1.0,
        inner_max_loop=1,
        max_outer_loop=20,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    print(np.unique(y_train))
    print(np.unique(y_train_pred))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    visualize_classification_prediction(learner, x_train, y_train, epoch=0, marker_size=20)


def test_bank_svmguide1():
    data_name = 'svmguide1'
    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
    test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')
    if not os.path.exists(test_file_name):
        raise Exception('File test not found')

    n_features = 4
    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    learner = BaNK(
        gamma=1.0/1.25727,
        rf_dim=384,
        inner_regularization=0.125,
        outer_regularization=0.125,
        alpha=10.0,
        kappa=0.1,
        inner_max_loop=1,
        max_outer_loop=200,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_bank_musk():
    np.seterr(under='warn')
    data_name = 'musk'
    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
    test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')
    if not os.path.exists(test_file_name):
        raise Exception('File test not found')

    n_features = 166
    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    learner = BaNK(
        gamma=1.0/138,
        rf_dim=384,
        inner_regularization=976.563,
        outer_regularization=128,
        alpha=10,
        kappa=0.1,
        inner_max_loop=1,
        max_outer_loop=201,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_bank_phishing():
    np.seterr(under='warn')
    data_name = 'phishing'
    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
    test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')
    if not os.path.exists(test_file_name):
        raise Exception('File test not found')

    n_features = 68
    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    learner = BaNK(
        gamma=1.0/1.06662,
        rf_dim=384,
        inner_regularization=0.00195313,
        outer_regularization=0.5,
        alpha=10,
        kappa=0.1,
        inner_max_loop=1,
        max_outer_loop=201,
        batch_size=5
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

if __name__ == '__main__':
    # pytest.main([__file__])
    # test_bank_2d()
    # test_bank_svmguide1()
    # test_bank_musk()
    test_bank_phishing()
