from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pytest
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male import TensorFlowModel
from male.models.linear import TensorFlowGLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_logit():
    print("========== Test TensorFlowGLM for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = TensorFlowGLM(model_name="TensorFlowGLM_logit",
                        l1_penalty=0.0,
                        l2_penalty=0.0,
                        num_epochs=10,
                        random_state=random_seed())

    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_softmax():
    print("========== Test TensorFlowGLM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = TensorFlowGLM(model_name="TensorFlowGLM_softmax",
                        link='softmax',
                        loss='softmax',
                        num_epochs=10,
                        random_state=random_seed())

    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_logit_gridsearch():
    print("========== Tune parameters for TensorFlowGLM for binary classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = TensorFlowGLM(model_name="TensorFlowGLM_logit_gridsearch",
                        num_epochs=10,
                        catch_exception=True,
                        random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = 1.0 - best_clf.score(x_train, y_train)
    test_err = 1.0 - best_clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)
    assert abs(test_err - (1.0 - gs.best_score_)) < 1e-4


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_softmax_gridsearch():
    print("========== Tune parameters for TensorFlowGLM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = TensorFlowGLM(model_name="TensorFlowGLM_softmax_gridsearch",
                        link='softmax',
                        loss='softmax',
                        num_epochs=10,
                        catch_exception=True,
                        random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = 1.0 - best_clf.score(x_train, y_train)
    test_err = 1.0 - best_clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)
    assert abs(test_err - (1.0 - gs.best_score_)) < 1e-4


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_regression():
    print("========== Test TensorFlowGLM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = TensorFlowGLM(model_name="TensorFlowGLM_regression",
                        task='regression',
                        link='linear',  # link function
                        loss='quadratic',  # loss function
                        l2_penalty=0.0,  # ridge regularization
                        l1_penalty=0.0,  # Lasso regularization
                        l1_smooth=1E-5,  # smoothing for Lasso regularization
                        l1_method='pseudo_huber',  # approximation method for L1-norm
                        learning_rate=0.001,
                        random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = - clf.score(x_train, y_train)
    test_err = - clf.score(x_test, y_test)
    print("Training MSE = %.4f" % train_err)
    print("Testing MSE = %.4f" % test_err)


@pytest.mark.skip(reason="There are exceptions that could not be catched")
def test_tfglm_regression_gridsearch():
    print("========== Tune parameters for TensorFlowGLM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'l1_penalty': [0.0, 0.0001],
              'l2_penalty': [0.0001, 0.001, 0.01]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = TensorFlowGLM(model_name="TensorFlowGLM_regression_gridsearch",
                        task='regression',
                        link='linear',  # link function
                        loss='quadratic',  # loss function
                        l2_penalty=0.0,  # ridge regularization
                        l1_penalty=0.0,  # Lasso regularization
                        l1_smooth=1E-5,  # smoothing for Lasso regularization
                        l1_method='pseudo_huber',  # approximation method for L1-norm
                        learning_rate=0.0001,
                        catch_exception=True,
                        random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best MSE {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = - best_clf.score(x_train, y_train)
    test_err = - best_clf.score(x_test, y_test)
    print("Training MSE = %.4f" % train_err)
    print("Testing MSE = %.4f" % test_err)
    assert abs(test_err + gs.best_score_) < 1e-4


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_cv(show=False, block_figure_on_end=False):
    print("========== Test cross-validation for TensorFlowGLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/TensorFlowGLM/iris_{epoch:04d}_{val_err:.6f}.pkl")
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

    clf = TensorFlowGLM(model_name="TensorFlowGLM_softmax_cv",
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


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_tfglm_save_load(show=False, block_figure_on_end=False):
    print("========== Test Save and Load functions for TensorFlowGLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=5, verbose=1)
    filepath = os.path.join(model_dir(), "male/TensorFlowGLM/iris_{epoch:04d}_{val_err:.6f}.pkl")
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

    clf = TensorFlowGLM(model_name="TensorFlowGLM_softmax_cv",
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

    save_file_path = os.path.join(model_dir(), "male/TensorFlowGLM/saved_model.ckpt")
    clf.save(file_path=save_file_path)
    clf1 = TensorFlowModel.load_model(save_file_path)

    clf1.num_epochs = 10
    clf1.fit(x, y)

    train_err = 1.0 - clf1.score(x_train, y_train)
    test_err = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_tfglm_logit()
    # test_tfglm_softmax()
    # test_tfglm_logit_gridsearch()
    # test_tfglm_softmax_gridsearch()
    # test_tfglm_regression()
    # test_tfglm_regression_gridsearch()
    # test_tfglm_cv(show=False, block_figure_on_end=True)
    # test_tfglm_save_load(show=True, block_figure_on_end=True)
