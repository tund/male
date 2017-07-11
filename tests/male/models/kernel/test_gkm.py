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
from male.models.kernel.gkm import GKM
from male.callbacks import Display


def test_gkm_visualization_2d(block_figure_on_end=False):
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

    learner = GKM(
        model_name="GKM",
        mode='batch',
        unlabel=-128,
        trade_off_1=0.1,
        trade_off_2=0.1,
        gamma=10,
        loss_func=GKM.HINGE,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        insensitive_epsilon=0.001,
        unlabel_loss_func_degree=2,
        sim_func=None,
        sim_params=(1.0, 0),
        callbacks=[predict_display],
        random_state=random_seed()
    )

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
