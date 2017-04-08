from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

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
from male.models.kernel import RKS
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_rks_check_grad():
    print("========== Check gradients ==========")
    np.random.seed(random_seed())

    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = RKS(model_name="checkgrad_RKS_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                loss='hinge',
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps

    model = RKS(model_name="checkgrad_RKS_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                loss='logit',
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = RKS(model_name="checkgrad_RKS_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                loss='hinge',
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps

    model = RKS(model_name="checkgrad_RKS_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                loss='logit',
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L1 loss">
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = RKS(model_name="checkgrad_RKS_l1",
                D=4,
                lbd=0.01,
                loss='l1',
                gamma=0.125,
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with L2 loss">
    model = RKS(model_name="checkgrad_RKS_l2",
                D=4,
                lbd=0.01,
                loss='l2',
                gamma=0.125,
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression with eps-insensitive loss">
    model = RKS(model_name="checkgrad_RKS_eps",
                D=4,
                lbd=0.01,
                loss='eps_insensitive',
                gamma=0.125,
                learning_rate=0.001)
    assert model.check_grad(x, y) < eps
    # </editor-fold>


def test_rks_bin():
    print("========== Test RKS for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = RKS(model_name="RKS_hinge",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_rks_softmax():
    print("========== Test RKS for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = RKS(model_name="RKS_hinge",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_rks_softmax_gridsearch():
    print("========== Tune parameters for RKS for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.03, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = RKS(model_name="RKS_hinge",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = 1.0 - best_clf.score(x_train, y_train)
    test_err = 1.0 - best_clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)
    assert abs(test_err - (1.0 - gs.best_score_)) < 1e-4


def test_rks_regression_gridsearch():
    print("========== Tune parameters for RKS for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'learning_rate': [0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = RKS(model_name="RKS_regression_gridsearch",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='l2',
              num_epochs=10,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = - best_clf.score(x_train, y_train)
    test_err = - best_clf.score(x_test, y_test)
    print("Training MSE = %.4f" % train_err)
    print("Testing MSE = %.4f" % test_err)
    assert abs(test_err + gs.best_score_) < 1e-4


def test_rks_cv():
    print("========== Test cross-validation for RKS ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=True)
    filepath = os.path.join(model_dir(), "male/RKS/irs_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    clf = RKS(model_name="RKS_hinge",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              learning_rate=0.1,
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


def test_rks_cv_gridsearch():
    print("========== Tune parameters for RKS including cross-validation ==========")

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
    filepath = os.path.join(model_dir(), "male/RKS/search/iris_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    clf = RKS(model_name="RKS_hinge",
              D=100,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              learning_rate=0.1,
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


def test_rks_syn2d_cv(block_figure_on_end=False):
    print("========== Test cross-validation for RKS on 2D data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_synthetic_2d()

    idx_train, idx_test = next(
        iter(StratifiedShuffleSplit(n_splits=1,
                                    test_size=40,
                                    random_state=random_seed()).split(x_train, y_train)))
    x0 = x_train[idx_train]
    y0 = y_train[idx_train]
    x1 = x_train[idx_test]
    y1 = y_train[idx_test]

    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(model_dir(),
                            "male/RKS/synthetic_2D_data_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)

    display = Display(layout=(3, 1),
                      dpi='auto',
                      block_on_end=block_figure_on_end,
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

    clf = RKS(model_name="syn2d_fogd_hinge",
              D=10,
              lbd=0.0,
              gamma=0.5,
              loss='hinge',
              num_epochs=10,
              batch_size=4,
              learning_rate=0.05,
              metrics=['loss', 'err'],
              callbacks=[display, early_stopping, checkpoint],
              cv=[-1] * x0.shape[0] + [0] * x1.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)

    train_err = 1 - clf.score(x_train, y_train)
    print("Training error = %.4f" % train_err)

    if block_figure_on_end:  # block_figure_on_end=False => using pytest, thus DON'T predict
        # save predictions
        y_test_pred = clf.predict(x_test)
        x_test[x_test == 0] = 1e-4
        dump_svmlight_file(x_test, y_test_pred,
                           os.path.join(data_dir(), "demo/synthetic_2D_data_test_predict"),
                           zero_based=False)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_rks_check_grad()
    # test_rks_bin()
    # test_rks_softmax()
    # test_rks_softmax_gridsearch()
    # test_rks_regression_gridsearch()
    # test_rks_cv()
    # test_rks_cv_gridsearch()
    # test_rks_syn2d_cv(block_figure_on_end=True)
