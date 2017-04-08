from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male import random_seed
from male.datasets import demo
from male.models.kernel import FOGD


def test_fogd_check_grad():
    print("========== Check gradients ==========")

    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = FOGD(model_name="checkgrad_FOGD_hinge",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 loss='hinge',
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps

    model = FOGD(model_name="checkgrad_FOGD_logit",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 loss='logit',
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = FOGD(model_name="checkgrad_FOGD_hinge",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 loss='hinge',
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps

    model = FOGD(model_name="checkgrad_FOGD_logit",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 loss='logit',
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L1 loss">
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = FOGD(model_name="checkgrad_FOGD_l1",
                 D=4,
                 lbd=0.01,
                 loss='l1',
                 gamma=0.125,
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L2 loss">
    model = FOGD(model_name="checkgrad_FOGD_l2",
                 D=4,
                 lbd=0.01,
                 loss='l2',
                 gamma=0.125,
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with eps-insensitive loss">
    model = FOGD(model_name="checkgrad_FOGD_eps",
                 D=4,
                 lbd=0.01,
                 loss='eps_insensitive',
                 gamma=0.125,
                 learning_rate=0.001)
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_fogd_bin():
    print("========== Test FOGD for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = FOGD(model_name="FOGD_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)

    # offline prediction
    print("Offline prediction")
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    train_err = 1 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_fogd_softmax():
    print("========== Test FOGD for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = FOGD(model_name="FOGD_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)

    # offline prediction
    print("Offline prediction")
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    train_err = 1 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_fogd_softmax_gridsearch():
    print("========== Tune parameters for FOGD for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.5, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = FOGD(model_name="FOGD_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               catch_exception=True,
               random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % best_clf.mistake)

    # offline prediction
    print("Offline prediction")
    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)
    train_err = 1 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_fogd_regression_gridsearch():
    print("========== Tune parameters for FOGD for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.5, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = FOGD(model_name="FOGD_l2",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='l2',
               catch_exception=True,
               random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % best_clf.mistake)

    # offline prediction
    print("Offline prediction")
    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)
    train_err = metrics.mean_squared_error(y_train, y_train_pred)
    test_err = metrics.mean_squared_error(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_fogd_check_grad()
    # test_fogd_bin()
    # test_fogd_softmax()
    # test_fogd_softmax_gridsearch()
    # test_fogd_regression_gridsearch()
