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

from male.configs import data_dir
from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male.models.kernel import KMM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def test_kmm_check_grad():
    print("========== Check gradients ==========")

    np.random.seed(random_seed())

    # <editor-fold desc="Binary classification using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_KMM_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Binary classification using MULTIPLE kernels">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_KMM_hinge",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_logit",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>s

    # <editor-fold desc="Multiclass classification using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_KMM_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification using MULTIPLE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_KMM_hinge",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_logit",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = KMM(model_name="checkgrad_KMM_l1",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l1',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_l2",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l2',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression using MULTIPLE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = KMM(model_name="checkgrad_KMM_l1",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='l1',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_l2",
                D=4,
                lbd=0.01,
                gamma=(0.125, 1.0),
                mode='batch',
                loss='l2',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_KMM_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                temperature=1.0,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_kmm_bin():
    print("========== Test KMM for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = KMM(model_name="KMM_hinge",
              D=4,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=2,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = KMM(model_name="KMM_hinge",
              D=4,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              num_kernels=4,
              temperature=0.1,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)


def test_kmm_softmax():
    print("========== Test KMM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = KMM(model_name="KMM_hinge",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = KMM(model_name="KMM_hinge",
              D=100,
              lbd=0.0,
              gamma=0.01,
              mode='online',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=random_seed(),
              verbose=1)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake)


def test_kmm_softmax_gridsearch():
    print("========== Tune parameters for KMM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'num_kernels': [1, 2, 4]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = KMM(model_name="KMM_hinge",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
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

    clf = KMM(model_name="KMM_hinge",
              D=100,
              lbd=0.01,
              gamma=0.01,
              mode='online',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)
    print("Mistake rate = %.4f" % best_clf.mistake)


def test_kmm_regression_gridsearch():
    print("========== Tune parameters for KMM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'gamma': [0.5, 1.0],
              'num_kernels': [1, 2, 4]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = KMM(model_name="KMM_l2",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='l2',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
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
    assert abs(test_err + gs.best_score_) < 1e-2

    clf = KMM(model_name="KMM_l2",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='online',
              loss='l2',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)
    print("Mistake rate = %.4f" % best_clf.mistake)
    assert abs(best_clf.mistake + gs.best_score_) < 1e-2


def test_kmm_cv():
    print("========== Test cross-validation for KMM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/KMM/iris_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    clf = KMM(model_name="KMM_hinge",
              D=20,
              lbd=0.0,
              gamma=0.1,
              mode='batch',
              loss='hinge',
              num_kernels=3,
              batch_size=100,
              temperature=1.0,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.1,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.1,
              learning_rate_alpha=0.1,
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


def test_kmm_cv_gridsearch():
    print("========== Tune parameters for KMM including cross-validation ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    params = {'gamma': [0.5, 1.0],
              'num_kernels': [1, 2, 4]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0]
                                   + [1] * x_test.shape[0])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    clf = KMM(model_name="KMM_hinge",
              D=20,
              lbd=0.0,
              gamma=0.1,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=1.0,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.1,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.1,
              learning_rate_alpha=0.1,
              metrics=['loss', 'err'],
              callbacks=[early_stopping],
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
    assert abs(test_err - (1 - gs.best_score_)) < 1e-4


def test_kmm_cv_disp(show=False, block_figure_on_end=False):
    print("========== Test cross-validation for KMM with Display ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/KMM/iris_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)
    display = Display(layout=(3, 1),
                      dpi='auto',
                      show=show,
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

    clf = KMM(model_name="KMM_hinge",
              D=20,
              lbd=0.0,
              gamma=0.1,
              mode='batch',
              loss='hinge',
              num_kernels=3,
              batch_size=100,
              temperature=1.0,
              num_epochs=10,
              num_nested_epochs=1,
              learning_rate=0.1,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.1,
              learning_rate_alpha=0.1,
              metrics=['loss', 'err'],
              callbacks=[display, early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_kmm_syn2d(show=False, block_figure_on_end=False):
    print("========== Test KMM on 2D data ==========")

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
    filepath = os.path.join(model_dir(), "male/KMM/syn2d_data_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    display = Display(layout=(3, 1),
                      dpi='auto',
                      show=show,
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

    clf = KMM(model_name="KMM_hinge",
              D=10,
              lbd=0.0,
              gamma=0.5,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=4,
              temperature=0.1,
              num_epochs=10,
              num_nested_epochs=0,
              learning_rate=0.001,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              metrics=['loss', 'err'],
              callbacks=[display, early_stopping, checkpoint],
              cv=[-1] * x0.shape[0] + [0] * x1.shape[0],
              random_state=random_seed())

    clf.fit(x, y)

    train_err = 1.0 - clf.score(x_train, y_train)
    print("Training error = %.4f" % train_err)

    if block_figure_on_end:
        # save predictions
        y_test_pred = clf.predict(x_test)
        x_test[x_test == 0] = 1e-4
        dump_svmlight_file(x_test, y_test_pred,
                           os.path.join(data_dir(), "demo/synthetic_2D_data_test_predict.libsvm"),
                           zero_based=False)


def test_kmm_syndata2(show=False, block_figure_on_end=False):
    print("========== Test KMM on Synthetic 2D data ==========")

    np.random.seed(random_seed())

    M = 1000  # number of samples
    D = 250  # number of random features (random feature dimension)
    x = 4 * np.random.randn(M)
    w = np.random.randn(2 * D)  # weights (\beta in BaNK paper)

    # <editor-fold desc="Example of 3 kernels with \mu different from 0">
    z = np.argmax(np.random.multinomial(1, [0.5, 0.25, 0.25], size=D), axis=1)
    o = np.hstack([np.pi / 2 + 0.5 * np.random.randn(D, 1), 0.25 * np.random.randn(D, 1),
                   np.pi + (1 / 3) * np.random.randn(D, 1)])
    omega = o[range(D), z]
    t = np.linspace(-7, 7, M)
    kt = (0.5 * np.exp(-0.125 * t * t) * np.cos(np.pi * t / 2) + 0.25 * np.exp(-(1 / 32) * t * t)
          + 0.25 * np.exp(-(1 / 18) * t * t) * np.cos(2 * np.pi * t / 2))
    # </editor-fold>

    # <editor-fold desc="Example of 3 kernels with \mu = 0">
    # z = np.argmax(np.random.multinomial(1, [0.5, 0.25, 0.25], size=D), axis=1)
    # o = np.hstack([0.5 * np.random.randn(D, 1), 0.25 * np.random.randn(D, 1),
    #                (1 / 3) * np.random.randn(D, 1)])
    # omega = o[range(D), z]
    # t = np.linspace(-7, 7, M)
    # kt = (0.5 * np.exp(-0.125 * t * t) + 0.25 * np.exp(-(1 / 32) * t * t)
    #       + 0.25 * np.exp(-(1 / 18) * t * t))
    # </editor-fold>

    phi = np.hstack([np.cos(np.outer(x, omega)) / np.sqrt(D),
                     np.sin(np.outer(x, omega)) / np.sqrt(D)])
    y = phi.dot(w) + np.random.randn(M)

    xx = t
    yy = np.zeros(xx.shape)
    phi_xx = np.hstack([np.cos(np.outer(xx, omega)) / np.sqrt(D),
                        np.sin(np.outer(xx, omega)) / np.sqrt(D)])
    phi_yy = np.hstack([np.cos(np.outer(yy, omega)) / np.sqrt(D),
                        np.sin(np.outer(yy, omega)) / np.sqrt(D)])
    approx_kt = np.sum(phi_xx * phi_yy, axis=1)

    loss_display = Display(layout=(2, 1),
                           dpi='auto',
                           show=show,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['err'],
                                     'type': 'line',
                                     'labels': ["training error"],
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    {'metrics': ['mu', 'gamma'],
                                     'type': 'line',
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    ])

    display = Display(layout=(1, 1),
                      dpi='auto',
                      show=show,
                      block_on_end=block_figure_on_end,
                      monitor=[{'metrics': ['syndata'],
                                'type': 'line',
                                'labels': ["Synthetic data"],
                                'title': "Synthetic data",
                                'xlabel': "t",
                                'ylabel': "k",
                                't': t,
                                'kt': kt,
                                'approx_kt': approx_kt,
                                },
                               ])

    # <editor-fold desc="1 kernel">
    # c = KMM(model_name="syndata2_kmm_l2",
    #         save_dir=os.path.join(HOME, "rmodel/pycs"),
    #         mode='batch', loss='l2', num_kernels=1, num_epochs=50,
    #         batch_size=100, lbd=0.0, learning_rate=0.1, learning_rate_mu=0.05,
    #         learning_rate_gamma=0.001, learning_rate_alpha=0.01,
    #         temperature=1.0, decay_rate=0.95, D=D, gamma=0.5, random_state=random_seed())
    # c.fit(x, y)
    # print("pi = {}".format(c.get_pi()))
    # print("mu = {}".format(c.mu_.T))
    # print("sigma = {}".format(np.exp(c.gamma_.T)))
    # print("alpha = {}".format(c.alpha_))
    #
    # phi_xxx = c.get_phi(xx[:, np.newaxis])
    # phi_yyy = c.get_phi(yy[:, np.newaxis])
    # approx_ktt = np.sum(phi_xxx * phi_yyy, axis=1)
    # plt.plot(t, approx_ktt, 'g:', linewidth=3, label='KMM-1')
    # </editor-fold>

    # <editor-fold desc="2 kernels">
    # c = KMM(model_name="syndata2_kmm_l2",
    #         save_dir=os.path.join(HOME, "rmodel/pycs"),
    #         mode='batch', loss='l2', num_kernels=2, num_epochs=50,
    #         batch_size=100, lbd=0.0, learning_rate=0.1, learning_rate_mu=0.05,
    #         learning_rate_gamma=0.001, learning_rate_alpha=0.01,
    #         temperature=1.0, decay_rate=0.95, D=D, gamma=0.5, random_state=random_seed())
    # c.fit(x, y)
    # print("pi = {}".format(c.get_pi()))
    # print("mu = {}".format(c.mu_.T))
    # print("sigma = {}".format(np.exp(c.gamma_.T)))
    # print("alpha = {}".format(c.alpha_))
    #
    # phi_xxx = c.get_phi(xx[:, np.newaxis])
    # phi_yyy = c.get_phi(yy[:, np.newaxis])
    # approx_ktt = np.sum(phi_xxx * phi_yyy, axis=1)
    # plt.plot(t, approx_ktt, 'y--', linewidth=3, label='KMM-2')
    # </editor-fold>

    # 3 kernels
    c = KMM(model_name="syndata2_kmm_l2",
            D=D,
            lbd=0.0,
            loss='l2',
            gamma=(0.1, 1.0, 2.0),
            mode='batch',
            num_kernels=3,
            num_epochs=10,
            batch_size=10,
            temperature=0.05,
            alternative_update=True,
            num_nested_epochs=0,
            adam_update=False,
            learning_rate=0.001,
            learning_rate_mu=0.01,
            learning_rate_gamma=0.01,
            learning_rate_alpha=0.01,
            metrics=['loss', 'err'],
            callbacks=[display, loss_display],
            random_state=random_seed())

    c.fit(x, y)

    print("pi = {}".format(c._get_pi()))
    print("mu = {}".format(c.mu.T))
    print("sigma = {}".format(np.exp(c.gamma_.T)))
    print("alpha = {}".format(c.alpha))

    # phi_xxx = c._get_phi(xx[:, np.newaxis])
    # phi_yyy = c._get_phi(yy[:, np.newaxis])
    # approx_ktt = np.sum(phi_xxx * phi_yyy, axis=1)
    # plt.plot(t, approx_ktt, 'b-', linewidth=3, label='KMM-3')

    # <editor-fold desc="Best">
    # c = KMM(model_name="syndata2_kmm_l2",
    #         save_dir="C:/Users/tund/rmodel/pycs",
    #         mode='batch', loss='l2', num_kernels=3, num_epochs=100, num_nested_epochs=100,
    #         batch_size=100, lbd=0.0, learning_rate=0.1, learning_rate_mu=0.05,
    #         learning_rate_gamma=0.001, learning_rate_alpha=0.01,
    #         temperature=1.0, D=D, gamma=0.5, random_state=random_seed())
    # </editor-fold>

    # <editor-fold desc="4 kernels">
    # c = KMM(model_name="syndata2_kmm_l2",
    #         save_dir=os.path.join(HOME, "rmodel/pycs"),
    #         mode='batch', loss='l2', num_kernels=4, num_epochs=50,
    #         batch_size=100, lbd=0.0, learning_rate=0.1, learning_rate_mu=0.05,
    #         learning_rate_gamma=0.001, learning_rate_alpha=0.01,
    #         temperature=1.0, decay_rate=0.95, D=D, gamma=0.5, random_state=random_seed())
    # c.fit(x, y)
    # print("pi = {}".format(c.get_pi()))
    # print("mu = {}".format(c.mu_.T))
    # print("sigma = {}".format(np.exp(c.gamma_.T)))
    # print("alpha = {}".format(c.alpha_))
    #
    # phi_xxx = c.get_phi(xx[:, np.newaxis])
    # phi_yyy = c.get_phi(yy[:, np.newaxis])
    # approx_ktt = np.sum(phi_xxx * phi_yyy, axis=1)
    # plt.plot(t, approx_ktt, 'm:', linewidth=3, label='KMM-4')
    # </editor-fold>

    # plt.xlabel('t')
    # plt.ylabel('k')
    # plt.legend()
    # plt.show()
    # plt.savefig("C:/Users/tund/Dropbox/sharing/DASCIMAL-CORE/Publications/2017-ICML-NonparamKernel/figs/syndata_approx.pdf",
    #             format='pdf', bbox_inches='tight')
    # plt.close()

    # Show synthetic data
    # y_pred = c.predict(x)
    # plt.figure()
    # plt.scatter(x, y_pred)
    # plt.xlabel('x')
    # plt.ylabel('y_pred')
    # plt.show()
    # plt.savefig("C:/Users/tund/Dropbox/sharing/DASCIMAL-CORE/Publications/2017-ICML-NonparamKernel/figs/syndata_xy.pdf", format='pdf', bbox_inches='tight')
    # plt.close()
    # print("RMSE = %.4f" % np.sqrt(-c.score(x, y)))
    # plt.show()


if __name__ == '__main__':
    pytest.main([__file__])
    # test_kmm_check_grad()
    # test_kmm_bin()
    # test_kmm_softmax()
    # test_kmm_softmax_gridsearch()
    # test_kmm_regression_gridsearch()
    # test_kmm_cv()
    # test_kmm_cv_gridsearch()
    # test_kmm_cv_disp(show=True, block_figure_on_end=True)
    # test_kmm_syn2d(show=True, block_figure_on_end=True)
    # test_kmm_syndata2(show=True, block_figure_on_end=True)
