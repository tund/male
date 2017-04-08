from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male import model_dir
from male import random_seed
from male.datasets import demo
from male.models.kernel import RRF
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_rrf_check_grad():
    print("========== Check gradients ==========")

    np.random.seed(random_seed())

    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = RRF(model_name="checkgrad_RRF_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = RRF(model_name="checkgrad_RRF_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = RRF(model_name="checkgrad_RRF_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = RRF(model_name="checkgrad_RRF_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L1 loss">
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = RRF(model_name="checkgrad_RRF_l1",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l1',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L2 loss">
    model = RRF(model_name="checkgrad_RRF_l2",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l2',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with eps-insensitive loss">
    model = RRF(model_name="checkgrad_RRF_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                learning_rate=0.001,
                learning_rate_gamma=0.001,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_rrf_bin():
    print("========== Test RRF for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)


def test_rrf_softmax():
    print("========== Test RRF for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)


def test_rrf_softmax_gridsearch():
    print("========== Tune parameters for RRF for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1.0 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)
    train_err = 1.0 - metrics.accuracy_score(y_train, y_train_pred)
    test_err = 1.0 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)
    print("Mistake rate = %.4f" % best_clf.mistake)
    assert abs(best_clf.mistake + gs.best_score_) < 1e-6


def test_rrf_regression_gridsearch():
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

    clf = RRF(model_name="RRF_l2",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='l2',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)
    train_err = metrics.mean_squared_error(y_train, y_train_pred)
    test_err = metrics.mean_squared_error(y_test, y_test_pred)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = RRF(model_name="RRF_l2",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='l2',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)
    print("Mistake rate = %.4f" % best_clf.mistake)
    assert abs(best_clf.mistake + gs.best_score_) < 1e-6


def test_rrf_cv():
    print("========== Test cross-validation for RRF ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2)
    filepath = os.path.join(model_dir(), "male/RRF/iris_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_rrf_cv_gridsearch():
    print("========== Tune parameters for RRF including cross-validation ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.05, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0]
                                   + [1] * x_test.shape[0])

    early_stopping = EarlyStopping(monitor='val_err', patience=2)
    filepath = os.path.join(model_dir(), "male/RRF/search/mnist_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    clf = RRF(model_name="RRF_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_epochs=10,
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)

    best_clf.fit(np.vstack([x_train, x_test]),
                 np.concatenate([y_train, y_test]))

    train_err = 1.0 - best_clf.score(x_train, y_train)
    test_err = 1.0 - best_clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)
    assert abs(test_err - (1.0 - gs.best_score_)) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__])
    # test_rrf_check_grad()
    # test_rrf_bin()
    # test_rrf_softmax()
    # test_rrf_softmax_gridsearch()
    # test_rrf_regression_gridsearch()
    # test_rrf_cv()
    # test_rrf_cv_gridsearch()
