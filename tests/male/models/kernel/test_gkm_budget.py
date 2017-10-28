from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

import numpy as np

from male.datasets import demo
from male.models.kernel.gkm_budget import GKM_BUDGET
from male.callbacks import Display


def test_gkm_budget_visualization_2d(show=False, block_figure_on_end=False):
    (x_train, y_train), (_, _) = demo.load_synthetic_2d_semi()

    print('num_samples: {}'.format(x_train.shape[0]))

    predict_display = Display(
        freq=100,
        dpi='auto',
        show=show,
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['predict'],
                  'title': "Visualization",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  'grid_size': 20,
                  'marker_size': 10,
                  'left': None,
                  'right': None,
                  'top': None,
                  'bottom': None
                  }]
    )

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
        sim_params=0.01,
        num_epochs=1,
        callbacks=[predict_display],
    )

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(
        y_train[y_train != -128], y_train_pred[y_train != -128])))


def run_gridsearch_a1a_u70():
    print("========== Tune parameters for IGKOL for classification ==========")

    (x_total, y_total), (x_test, y_test) = demo.load_a1a_semi()

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
        'trade_off_1': 2.0 ** np.arange(-3, 5, 6),
        'trade_off_2': 2.0 ** np.arange(-3, 5, 6)
    }

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_valid.shape[0])

    clf = GKM_BUDGET(
        model_name="GKM_BUDGET",
        mode='batch',
        unlabel=-1,
        trade_off_1=1,
        trade_off_2=1,
        gamma=100.0,
        loss_func=GKM_BUDGET.S_HINGE,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        insensitive_epsilon=0.001,
        unlabel_loss_func_degree=1,
        sim_func=None,
        sim_params=0.01,
        num_epochs=1,
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
    # test_gkm_budget_visualization_2d(True, True)
    # run_gridsearch_a1a_u70()
