from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male import Model
from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male.optimizers import SGD
from male.models.linear import GLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_glm_check_grad():
    print("========== Check gradients ==========")
    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = GLM(model_name="checkgrad_GLM_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = GLM(model_name="checkgrad_GLM_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multilabel classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.empty(num_data, dtype=np.object)
    for i in range(num_data):
        y[i] = tuple(np.random.choice(num_classes, np.random.randint(num_classes) + 1,
                                      replace=False))

    model = GLM(model_name="checkgrad_GLM_multilogit",
                task='multilabel',
                link='logit',  # link function
                loss='multilogit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_multilogit",
                task='multilabel',
                link='logit',  # link function
                loss='multilogit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_multilogit",
                task='multilabel',
                link='logit',  # link function
                loss='multilogit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_multilogit",
                task='multilabel',
                link='logit',  # link function
                loss='multilogit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression">
    eps = 1e-6
    num_data = 10
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    model = GLM(model_name="checkgrad_GLM_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_GLM_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=random_seed())
    assert model.check_grad(x, y) < eps
    # </editor-fold>


def test_glm_logit():
    print("========== Test GLM for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = GLM(model_name="GLM_logit",
              l1_penalty=0.0,
              l2_penalty=0.0,
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = GLM(model_name="GLM_logit",
              optimizer='sgd',
              l1_penalty=0.0,
              l2_penalty=0.0,
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    optz = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    clf = GLM(model_name="GLM_logit",
              optimizer=optz,
              l1_penalty=0.0,
              l2_penalty=0.0,
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_glm_softmax():
    print("========== Test GLM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = GLM(model_name="GLM_softmax",
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf = GLM(model_name="GLM_softmax",
              optimizer='sgd',
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_glm_multilogit():
    print("========== Test GLM for multilabel classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_yeast()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = GLM(model_name="GLM_multilogit",
              task='multilabel',
              link='logit',
              loss='multilogit',
              random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_f1 = clf.score(x_train, y_train)
    test_f1 = clf.score(x_test, y_test)
    print("Training weighted-F1-macro = %.4f" % train_f1)
    print("Testing weighted-F1-macro = %.4f" % test_f1)

    optz = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    clf = GLM(model_name="GLM_multilogit",
              task='multilabel',
              optimizer=optz,
              link='logit',
              loss='multilogit',
              random_state=random_seed(),
              verbose=1)

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_f1 = clf.score(x_train, y_train)
    test_f1 = clf.score(x_test, y_test)
    print("Training weighted-F1-macro = %.4f" % train_f1)
    print("Testing weighted-F1-macro = %.4f" % test_f1)


def test_glm_logit_gridsearch():
    print("========== Tune parameters for GLM for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="GLM_logit_gridsearch",
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


def test_glm_softmax_gridsearch():
    print("========== Tune parameters for GLM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="GLM_softmax_gridsearch",
              link='softmax',
              loss='softmax',
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


def test_glm_multilogit_gridsearch():
    print("========== Tune parameters for GLM for multilabel classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_yeast()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="GLM_multilogit_gridsearch",
              task='multilabel',
              link='logit',
              loss='multilogit',
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best weighted-F1-macro {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_f1 = best_clf.score(x_train, y_train)
    test_f1 = best_clf.score(x_test, y_test)
    print("Training weighted-F1-macro = %.4f" % train_f1)
    print("Testing weighted-F1-macro = %.4f" % test_f1)
    assert abs(test_f1 - gs.best_score_) < 1e-4


def test_glm_regression():
    print("========== Test GLM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = GLM(model_name="GLM_regression",
              task='regression',
              link='linear',  # link function
              loss='quadratic',  # loss function
              l2_penalty=0.0,  # ridge regularization
              l1_penalty=0.0,  # Lasso regularization
              l1_smooth=1E-5,  # smoothing for Lasso regularization
              l1_method='pseudo_huber',  # approximation method for L1-norm
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = - clf.score(x_train, y_train)
    test_err = - clf.score(x_test, y_test)
    print("Training MSE = %.4f" % train_err)
    print("Testing MSE = %.4f" % test_err)


def test_glm_regression_gridsearch():
    print("========== Tune parameters for GLM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="GLM_regression_gridsearch",
              task='regression',
              link='linear',  # link function
              loss='quadratic',  # loss function
              l2_penalty=0.0,  # ridge regularization
              l1_penalty=0.0,  # Lasso regularization
              l1_smooth=1E-5,  # smoothing for Lasso regularization
              l1_method='pseudo_huber',  # approximation method for L1-norm
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


def test_glm_cv(show=False, block_figure_on_end=False):
    print("========== Test cross-validation for GLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/glm/iris_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
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

    weight_display = Display(title="Filters",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(6, 15),
                             freq=1,
                             show=show,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'disp_dim': (2, 2),
                                       'tile_shape': (3, 1),
                                       },
                                      ])

    clf = GLM(model_name="GLM_softmax_cv",
              link='softmax',
              loss='softmax',
              optimizer='sgd',
              num_epochs=20,
              batch_size=10,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint, loss_display, weight_display],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_glm_save_load(show=False, block_figure_on_end=False):
    print("========== Test Save and Load functions for GLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=5, verbose=1)
    filepath = os.path.join(model_dir(), "male/GLM/iris_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
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

    weight_display = Display(title="Filters",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(6, 15),
                             freq=1,
                             show=show,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'disp_dim': (2, 2),
                                       'tile_shape': (3, 1),
                                       },
                                      ])

    clf = GLM(model_name="GLM_softmax_cv",
              link='softmax',
              loss='softmax',
              optimizer='sgd',
              num_epochs=4,
              batch_size=10,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint, loss_display, weight_display],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=random_seed(),
              verbose=1)

    clf.fit(x, y)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    save_file_path = os.path.join(model_dir(), "male/GLM/saved_model.pkl")
    clf.save(file_path=save_file_path)
    clf1 = Model.load_model(save_file_path)
    clf1.num_epochs = 10
    clf1.fit(x, y)

    train_err = 1.0 - clf1.score(x_train, y_train)
    test_err = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_glm_check_grad()
    # test_glm_logit()
    # test_glm_softmax()
    # test_glm_multilogit()
    # test_glm_logit_gridsearch()
    # test_glm_softmax_gridsearch()
    # test_glm_multilogit_gridsearch()
    # test_glm_regression()
    # test_glm_regression_gridsearch()
    # test_glm_cv(show=True, block_figure_on_end=True)
    # test_glm_save_load(show=True, block_figure_on_end=True)
