from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedShuffleSplit

from male import data_dir
from male import model_dir
from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.kernel import FOGD
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_fogd_check_grad():
    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = FOGD(model_name="checkgrad_fogd_hinge",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 mode='batch',
                 loss='hinge',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = FOGD(model_name="checkgrad_fogd_logit",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 mode='batch',
                 loss='logit',
                 learning_rate=0.001)
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

    model = FOGD(model_name="checkgrad_fogd_hinge",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 mode='batch',
                 loss='hinge',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = FOGD(model_name="checkgrad_fogd_logit",
                 D=4,
                 lbd=0.01,
                 gamma=0.125,
                 mode='batch',
                 loss='logit',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L1 loss">
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = FOGD(model_name="checkgrad_fogd_l1",
                 D=4,
                 lbd=0.01,
                 loss='l1',
                 gamma=0.125,
                 mode='batch',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L2 loss">
    model = FOGD(model_name="checkgrad_fogd_l2",
                 D=4,
                 lbd=0.01,
                 loss='l2',
                 gamma=0.125,
                 mode='batch',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with eps-insensitive loss">
    model = FOGD(model_name="checkgrad_fogd_eps",
                 D=4,
                 lbd=0.01,
                 loss='eps_insensitive',
                 gamma=0.125,
                 mode='batch',
                 learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_fogd_mnist_bin():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    idx_train = np.where(np.uint8(y_train == 0) | np.uint8(y_train == 1))[0]
    print("# training samples = {}".format(len(idx_train)))
    x_train /= 255.0
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    idx_test = np.where(np.uint8(y_test == 0) | np.uint8(y_test == 1))[0]
    print("# testing samples = {}".format(len(idx_test)))
    x_test /= 255.0
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=100,
               random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='online',
               random_state=6789)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_)


def test_fogd_mnist_softmax():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=100,
               random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='online',
               num_epochs=100,
               random_state=6789)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_)


def test_fogd_mnist_softmax_gridsearch():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=10,
               random_state=6789)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_fogd_regression_gridsearch():
    np.random.seed(random_seed())

    # regression
    eps = 1e-6
    num_data = 100
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * 70 + [1] * 30)

    clf = FOGD(model_name="mnist_fogd_l2",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='l2',
               mode='batch',
               num_epochs=10,
               random_state=6789)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x[:70], y[:70])

    y_train_pred = best_clf.predict(x[:70])
    y_test_pred = best_clf.predict(x[70:])

    print("Training error = %.4f" % (metrics.mean_squared_error(y[:70], y_train_pred)))
    print("Testing error = %.4f" % (metrics.mean_squared_error(y[70:], y_test_pred)))


def test_fogd_mnist_cv():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    filepath = os.path.join(model_dir(), "male/fogd/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=100,
               learning_rate=0.001,
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


def test_fogd_mnist_cv_gridsearch():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
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
    filepath = os.path.join(model_dir(), "male/fogd/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)

    clf = FOGD(model_name="mnist_fogd_hinge",
               D=100,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=100,
               learning_rate=0.1,
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


def test_fogd_syn2d_cv():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_synthetic_2d()

    # idx_train = np.random.permutation(x_train.shape[0])
    # x_train = x_train[idx_train]
    # y_train = y_train[idx_train]

    idx_train, idx_test = next(
        iter(StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=6789).split(x_train,
                                                                                       y_train))
    )
    x0 = x_train[idx_train]
    y0 = y_train[idx_train]
    x1 = x_train[idx_test]
    y1 = y_train[idx_test]

    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    filepath = os.path.join(model_dir(),
                            "male/fogd/synthetic_2D_data_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    display = Display(layout=(3, 1),
                      dpi='auto',
                      monitor=[{'metrics': ['loss', 'val_loss'],
                                'type': 'line',
                                'labels': ["training loss", "validation loss"],
                                'title': "Learning losses",
                                'xlabel': "epoch",
                                'ylabel': "loss",
                                },
                               {'metrics': ['err', 'val_err'],
                                'type': 'line',
                                'title': "Learning errors",
                                'xlabel': "epoch",
                                'ylabel': "error",
                                },
                               {'metrics': ['err'],
                                'type': 'line',
                                'labels': ["training error"],
                                'title': "Learning errors",
                                'xlabel': "epoch",
                                'ylabel': "error",
                                },
                               ])

    clf = FOGD(model_name="syn2d_fogd_hinge",
               D=10,
               lbd=0.0,
               gamma=0.5,
               loss='hinge',
               mode='batch',
               num_epochs=1000,
               batch_size=4,
               learning_rate=0.05,
               metrics=['loss', 'err'],
               callbacks=[display, early_stopping, checkpoint],
               cv=[-1] * x0.shape[0] + [0] * x1.shape[0],
               random_state=6789,
               verbose=1)

    clf.fit(x, y)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))

    # save predictions
    x_test[x_test == 0] = 1e-4
    dump_svmlight_file(x_test, y_test_pred,
                       os.path.join(data_dir(), "demo/synthetic_2D_data_test_predict"),
                       zero_based=False)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_fogd_check_grad()
    # test_fogd_mnist_bin()
    # test_fogd_mnist_softmax()
    # test_fogd_mnist_softmax_gridsearch()
    # test_fogd_regression_gridsearch()
    # test_fogd_mnist_cv()
    # test_fogd_mnist_cv_gridsearch()
    # test_fogd_syn2d_cv()
