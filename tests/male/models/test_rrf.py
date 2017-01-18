from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male.models.kernel import RRF
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_rrf_check_grad():
    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = RRF(model_name="checkgrad_rrf_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = RRF(model_name="checkgrad_rrf_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
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

    model = RRF(model_name="checkgrad_rrf_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = RRF(model_name="checkgrad_rrf_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L1 loss">
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = RRF(model_name="checkgrad_rrf_l1",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l1',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L2 loss">
    model = RRF(model_name="checkgrad_rrf_l2",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l2',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with eps-insensitive loss">
    model = RRF(model_name="checkgrad_rrf_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                learning_rate=0.001,
                learning_rate_gamma=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_rrf_mnist_bin():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    idx_train = np.where(np.uint8(y_train == 0) | np.uint8(y_train == 1))[0]
    print("# training samples = {}".format(len(idx_train)))
    x_train = x_train.toarray() / 255.0
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    idx_test = np.where(np.uint8(y_test == 0) | np.uint8(y_test == 1))[0]
    print("# testing samples = {}".format(len(idx_test)))
    x_test = x_test.toarray() / 255.0
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_)


def test_rrf_mnist_softmax():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='online',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_)


def test_rrf_mnist_softmax_gridsearch():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_rrf_regression_gridsearch():
    # regression
    eps = 1e-6
    num_data = 100
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * 70 + [1] * 30)

    clf = RRF(model_name="regress_rrf_l2",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='l2',
              learning_rate=0.001,
              learning_rate_gamma=0.001)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x[:70], y[:70])

    y_train_pred = best_clf.predict(x[:70])
    y_test_pred = best_clf.predict(x[70:])

    print("Training error = %.4f" % (metrics.mean_squared_error(y[:70], y_train_pred)))
    print("Testing error = %.4f" % (metrics.mean_squared_error(y[70:], y_test_pred)))


def test_rrf_mnist_cv():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    filepath = os.path.join(HOME, "rmodel/male/rrf/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    clf.fit(x, y)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_rrf_mnist_cv_gridsearch():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0]
                                   + [1] * x_test.shape[0])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    filepath = os.path.join(HOME, "rmodel/male/rrf/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)

    clf = RRF(model_name="mnist_rrf_hinge",
              D=100,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              learning_rate=0.001,
              learning_rate_gamma=0.001,
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)

    best_clf.fit(np.vstack([x_train, x_test]),
                 np.concatenate([y_train, y_test]))

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    # pytest.main([__file__])
    test_rrf_check_grad()
    test_rrf_mnist_bin()
    test_rrf_mnist_softmax()
    # test_rrf_mnist_softmax_gridsearch()
    # test_rrf_regression_gridsearch()
    test_rrf_mnist_cv()
    # test_rrf_mnist_cv_gridsearch()
