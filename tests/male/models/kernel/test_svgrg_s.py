from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import pickle

from sklearn import metrics

from male import random_seed
from male.datasets import demo
from male.models.kernel.svrg_s import SVRG_S
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

    learner = SVRG_S(
        regular_param=0.01,
        learning_rate_scale=1.0,
        gamma=10,
        rf_dim=400,
        num_epochs=2,
        cache_size=6,
        freq_update_full_model=10,
        oracle=SVRG_S.BUDGET,
        core_max=10,
        coverage_radius=100.0,
        loss_func=SVRG_S.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        callbacks=[loss_display, predict_display],
        metrics=['train_loss', 'obj_func'],
        freq_calc_metrics=20,
        random_state=random_seed())

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))


def test_svmguide1(block_figure_on_end=False):
    print("========== Test SVRG_S on svmguide1 dataset ==========")

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

    learner = SVRG_S(
        regular_param=0.01,
        learning_rate_scale=0.7,
        gamma=2.0,
        rf_dim=400,
        num_epochs=1,
        cache_size=50,
        freq_update_full_model=100,
        oracle=SVRG_S.COVERAGE,
        core_max=100,
        coverage_radius=0.9,
        loss_func=SVRG_S.LOGISTIC,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        callbacks=[loss_display],
        metrics=['train_loss', 'obj_func'],
        freq_calc_metrics=10,
        random_state=random_seed()
    )

    learner.fit(x_train, y_train)

    y_train_pred = learner.predict(x_train)
    y_test_pred = learner.predict(x_test)
    print("Dataset: {}".format(data_name))
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Test error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))
    # pickle.dump(learner.history.history, open("log/igkol." + data_name + '.log.p', "wb"))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_svrg_s_visualization_2d(block_figure_on_end=True)
    # test_svmguide1()
