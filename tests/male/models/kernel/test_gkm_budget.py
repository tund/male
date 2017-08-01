from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import pickle
import os

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

import numpy as np

from male import random_seed
from male.datasets import demo
from male.common import data_dir
from male.models.kernel.gkm_budget import GKM_BUDGET
from male.callbacks import Display


def test_GKM_BUDGET_visualization_2d(block_figure_on_end=False):
    data_name = '2d.semi'
    n_features = 2
    train_file_name = os.path.join(data_dir(), data_name + '.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')

    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_train = x_train.toarray()

    print('num_samples: {}'.format(x_train.shape[0]))

    predict_display = Display(
        freq=100,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['predict'],
                  'title': "Visualization",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  'grid_size': 200,
                  'marker_size': 10,
                  'left': None,
                  'right': None,
                  'top': None,
                  'bottom': None
                  }]
    )

    loss_display = Display(
        freq=2,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['train_loss'],
                  'type': 'line',
                  'title': "Learning losses",
                  'xlabel': "data points",
                  'ylabel': "loss",
                  }]
    )

    np.seterr(under='ignore')
    learner = GKM_BUDGET(
        model_name="GKM_BUDGET",
        mode='batch',
        unlabel=-1,
        trade_off_1=1,
        trade_off_2=2,
        gamma=500.0,
        loss_func=GKM_BUDGET.S_HINGE,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        insensitive_epsilon=0.001,
        unlabel_loss_func_degree=1,
        sim_func=None,
        sim_params=(1.0, 0),
        num_epochs=20,
        callbacks=[predict_display],
        # metrics=['train_loss'],
        random_state=np.random.randint(100000)
    )

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(
        y_train[y_train != -128], y_train_pred[y_train != -128])))


def run_gridsearch_a1a_u70():
    print("========== Tune parameters for IGKOL for classification ==========")

    np.random.seed(random_seed())

    data_name = 'dla1a.txt'
    n_features = 123

    file_name = os.path.join(data_dir(), data_name + '_train.70u.libsvm')

    print(file_name)

    if not os.path.exists(file_name):
        raise Exception('File not found')

    x_total, y_total = load_svmlight_file(file_name, n_features=n_features)
    x_total = x_total.toarray()

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

    params = {
        'gamma': 2.0 ** np.arange(-3, 5, 2),
        'trade_off_1': 2.0 ** np.arange(-3, 5, 2),
        'trade_off_2': 2.0 ** np.arange(-3, 5, 2)
    }

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_valid.shape[0])

    clf = GKM_BUDGET(
        model_name="GKM_BUDGET",
        mode='batch',
        unlabel=-128,
        trade_off_1=1,
        trade_off_2=1,
        gamma=100.0,
        loss_func=GKM_BUDGET.S_HINGE,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        insensitive_epsilon=0.001,
        unlabel_loss_func_degree=1,
        sim_func=None,
        sim_params=(1.0, 0),
        num_epochs=10,
        random_state=np.random.randint(100000)
    )

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1.0 - gs.best_score_, gs.best_params_))

    # best param prediction
    print("Best param prediction")
    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_total, y_total)

    test_file_name = os.path.join(data_dir(), data_name + '_test.70u.libsvm')
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)
    x_test = x_test.toarray()

    y_total_pred = best_clf.predict(x_total)
    y_test_pred = best_clf.predict(x_test)
    total_err = 1 - metrics.accuracy_score(y_total, y_total_pred)
    test_err = 1 - metrics.accuracy_score(y_test, y_test_pred)
    print("Training error = %.4f" % total_err)
    print("Testing error = %.4f" % test_err)

if __name__ == '__main__':
    # pytest.main([__file__])
    test_GKM_BUDGET_visualization_2d(True)
    # run_gridsearch_a1a_u70()
