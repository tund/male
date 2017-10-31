from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from sklearn import metrics

from male.configs import random_seed
from male.datasets import demo
from male.models.kernel import DualSGD


def test_dualsgd_bin():
    print("========== Test DualSGD for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = DualSGD(model_name="DualSGD_hinge",
                  k=20,
                  D=200,
                  gamma=1.0,
                  lbd=3.3593684387335183e-05,
                  loss='hinge',
                  maintain='k-merging',
                  max_budget_size=100,
                  random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)
    print("Budget size = %d" % clf.budget_size)

    # offline prediction
    print("Offline prediction")
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    train_err = 1 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_dualsgd_softmax():
    print("========== Test DualSGD for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = DualSGD(model_name="DualSGD_hinge",
                  k=20,
                  D=200,
                  gamma=1.0,
                  lbd=3.3593684387335183e-05,
                  loss='hinge',
                  maintain='k-merging',
                  max_budget_size=100,
                  random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)
    print("Budget size = %d" % clf.budget_size)

    # offline prediction
    print("Offline prediction")
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    train_err = 1 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_dualsgd_regression():
    print("========== Test DualSGD for Regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    est = DualSGD(model_name="DualSGD_eps_insensitive",
                  k=20,
                  D=200,
                  gamma=1.0,
                  eps=0.001,
                  lbd=0.00128,
                  loss='eps_insensitive',
                  maintain='k-merging',
                  max_budget_size=100,
                  random_state=random_seed())
    est.fit(x_train, y_train)

    print("Mistake rate = %.4f" % est.mistake)
    print("Budget size = %d" % est.budget_size)

    # offline prediction
    print("Offline prediction")
    y_train_pred = est.predict(x_train)
    y_test_pred = est.predict(x_test)
    train_err = metrics.mean_squared_error(y_train, y_train_pred)
    test_err = metrics.mean_squared_error(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_dualsgd_bin()
    # test_dualsgd_softmax()
    # test_dualsgd_regression()
