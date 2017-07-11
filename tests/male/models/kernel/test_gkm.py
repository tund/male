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
from male.models.kernel.gkm import GKM
from male.callbacks import Display


def test_gkm_visualization_2d(block_figure_on_end=False):
    data_name = '2d.semi'
    n_features = 2
    train_file_name = os.path.join(data_dir(), data_name + '.libsvm')

    if not os.path.exists(train_file_name):
        raise Exception('File train not found')

    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_train = x_train.toarray()

    print('num_samples: {}'.format(x_train.shape[0]))

    predict_display = Display(
        freq=2000,
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
    learner = GKM(
        model_name="GKM",
        mode='batch',
        unlabel=-128,
        trade_off_1=1.0,
        trade_off_2=1.0,
        gamma=500.0,
        loss_func=GKM.HINGE,
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        insensitive_epsilon=0.001,
        unlabel_loss_func_degree=2,
        sim_func=None,
        sim_params=(1.0, 0),
        callbacks=[predict_display],
        # metrics=['train_loss'],
        random_state=random_seed()
    )

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))

if __name__ == '__main__':
    # pytest.main([__file__])
    test_gkm_visualization_2d(True)
