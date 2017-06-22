from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import pickle
import os

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

import numpy as np

from male import random_seed
from male.datasets import demo
from male.models.kernel.svrg import SVRG
from male.callbacks import Display


def test_svrg_s_visualization_2d(block_figure_on_end=False):
    (x_train, y_train), (_, _) = demo.load_synthetic_2d()

    print('num_samples: {}'.format(x_train.shape[0]))

    predict_display = Display(
        freq=1,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['predict'],
                  'title': "Visualization",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  'grid_size': 100,
                  'marker_size': 10,
                  'left': None,
                  'right': None,
                  'top': None,
                  'bottom': None
                  }]
    )

    loss_display = Display(
        freq=1,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['train_loss', 'obj_func'],
                  'type': 'line',
                  'title': "Learning losses",
                  'xlabel': "data points",
                  'ylabel': "loss",
                  }]
    )

    learner = SVRG(
        regular_param=0.01,
        learning_rate_scale=1.0,
        gamma=10,
        rf_dim=400,
        num_epochs=2,
        cache_size=6,
        freq_update_full_model=10,
        oracle=SVRG.COVERAGE,
        core_max=10,
        coverage_radius=100.0,
        loss_func=SVRG.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        callbacks=[loss_display, predict_display],
        metrics=['train_loss', 'obj_func'],
        freq_calc_metrics=20,
        random_state=random_seed())

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))


def check_grad():
    print("========== Check gradients ==========")
    np.random.seed(random_seed())

    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)
    print(np.unique(y))

    model = SVRG(
        regular_param=0.01,
        learning_rate_scale=1.0,
        gamma=10,
        rf_dim=400,
        num_epochs=2,
        cache_size=6,
        freq_update_full_model=10,
        oracle=SVRG.COVERAGE,
        core_max=10,
        coverage_radius=100.0,
        loss_func=SVRG.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        freq_calc_metrics=20,
    )

    model.fit(x, y)

    for n in range(num_data):
        assert model.check_grad(x[n, :], y[n]) < eps


def test_svmguide1(block_figure_on_end=False):
    print("========== Test SVRG on svmguide1 dataset ==========")

    data_name = 'svmguide1'

    (x_train, y_train), (x_test, y_test) = demo.load_svmguide1()

    loss_display = Display(
        freq=1,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['train_loss', 'obj_func'],
                  'type': 'line',
                  'title': "Learning losses",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  }]
    )

    print(x_train.shape)

    learner = SVRG(
        mode='online',
        regular_param=0.01,
        learning_rate_scale=0.7,
        gamma=2.0,
        rf_dim=400,
        num_epochs=1,
        cache_size=50,
        freq_update_full_model=100,
        oracle=SVRG.COVERAGE,
        core_max=100,
        coverage_radius=0.9,
        loss_func=SVRG.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        callbacks=[loss_display],
        metrics=['train_loss', 'obj_func'],
        freq_calc_metrics=1,
        random_state=random_seed()
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))
    # pickle.dump(learner.history.history, open("log/igkol." + data_name + '.log.p', "wb"))


def run_svmguide1_gridsearch():
    print("========== Tune parameters for SVRG for classification ==========")

    data_name = 'svmguide1'

    (x_total, y_total), (x_test, y_test) = demo.load_svmguide1()

    n_total = x_total.shape[0]
    percent = 0.8
    n_train = int(percent * n_total)
    idx_train = np.random.permutation(n_total)[:n_train]
    mask = np.zeros(n_total, dtype=bool)
    mask[idx_train] = True
    x_train = x_total[mask, :]
    x_valid = x_total[~mask, :]

    y_train = y_total[mask]
    y_valid = y_total[~mask]

    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_valid.shape[0]))

    x = np.vstack((x_train, x_valid))
    y = np.concatenate((y_train, y_valid))

    params = {'regular_param': [0.0001, 0.00001],
              'gamma': [0.25, 0.5, 1, 2]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_valid.shape[0])

    clf = SVRG(
        mode='batch',
        regular_param=0.01,
        learning_rate_scale=0.8,
        gamma=2.0,
        rf_dim=400,
        num_epochs=1,
        cache_size=50,
        freq_update_full_model=100,
        oracle=SVRG.COVERAGE,
        core_max=100,
        coverage_radius=0.9,
        loss_func=SVRG.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        random_state=random_seed()
    )

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1.0 - gs.best_score_, gs.best_params_))

    # best param prediction
    print("Best param prediction")
    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_total, y_total)

    y_total_pred = best_clf.predict(x_total)
    y_test_pred = best_clf.predict(x_test)
    total_err = 1 - metrics.accuracy_score(y_total, y_total_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % total_err)
    print("Testing error = %.4f" % test_err)

if __name__ == '__main__':
    pytest.main([__file__])
    # test_svrg_s_visualization_2d(block_figure_on_end=True)
    # check_grad()
    # test_svmguide1()
    # run_svmguide1_gridsearch()
