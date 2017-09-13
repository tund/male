from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint
from male.models.deep_learning.rbm import SemiSupervisedRBM


def test_ssrbm_classification(show_figure=False, block_figure_on_end=False):
    print("========== Test Semi-Supervised RBM for Classification ==========")

    num_labeled_data = 1000

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    # remove some labels
    idx_train, idx_test = next(
        iter(StratifiedShuffleSplit(
            n_splits=1, test_size=num_labeled_data, random_state=random_seed()).split(x_train, y_train)))
    y_train[idx_train] = 10 ** 8

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
                               show=show_figure,
                               block_on_end=block_figure_on_end,
                               monitor=[{'metrics': ['recon_err', 'val_recon_err'],
                                         'type': 'line',
                                         'labels': ["training recon error",
                                                    "validation recon error"],
                                         'title': "Reconstruction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        {'metrics': ['loss', 'val_loss'],
                                         'type': 'line',
                                         'labels': ["training loss", "validation loss"],
                                         'title': "Learning Losses",
                                         'xlabel': "epoch",
                                         'ylabel': "loss",
                                         },
                                        {'metrics': ['err', 'val_err'],
                                         'type': 'line',
                                         'labels': ["training error", "validation error"],
                                         'title': "Prediction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        # {'metrics': ['loglik_csl', 'val_loglik_csl'],
                                        #  'type': 'line',
                                        #  'labels': ["training loglik (CSL)", "validation loglik (CSL)"],
                                        #  'title': "Loglikelihoods using CSL",
                                        #  'xlabel': "epoch",
                                        #  'ylabel': "loglik",
                                        #  },
                                        ])

    filter_display = Display(title="Receptive Fields",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 15,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (3, 5),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/ssRBM/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SemiSupervisedRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        learning_rate=0.1,
        w_init=0.1,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='first_order',
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[filter_display, learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x, y)

    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())

    print("=========== Predicted by Semi-Supervised RBM ============")
    print("Train accuracy = {0:>1.4f}\tTest accuracy = {1:>1.4f}".format(
        accuracy_score(y_train, model.predict(x_train)),
        accuracy_score(y_test, model.predict(x_test))))

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(x_train1, y_train)

    print("=========== Predicted by kNN ============")
    print("Train accuracy = {0:>1.4f}\tTest accuracy = {1:>1.4f}".format(
        accuracy_score(y_train, clf.predict(x_train1)),
        accuracy_score(y_test, clf.predict(x_test1))))


def test_ssrbm_regression(show_figure=False, block_figure_on_end=False):
    print("========== Test Semi-Supervised RBM for Classification ==========")

    num_labeled_data = 100

    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()

    # remove some labels
    idx_train, idx_test = next(
        iter(ShuffleSplit(
            n_splits=1, test_size=num_labeled_data,
            random_state=random_seed()).split(x_train, y_train)))
    y_train[idx_train] = 10 ** 8

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
                               show=show_figure,
                               block_on_end=block_figure_on_end,
                               monitor=[{'metrics': ['recon_err', 'val_recon_err'],
                                         'type': 'line',
                                         'labels': ["training recon error",
                                                    "validation recon error"],
                                         'title': "Reconstruction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        {'metrics': ['loss', 'val_loss'],
                                         'type': 'line',
                                         'labels': ["training loss", "validation loss"],
                                         'title': "Learning Losses",
                                         'xlabel': "epoch",
                                         'ylabel': "loss",
                                         },
                                        {'metrics': ['err', 'val_err'],
                                         'type': 'line',
                                         'labels': ["training error", "validation error"],
                                         'title': "Prediction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        # {'metrics': ['loglik_csl', 'val_loglik_csl'],
                                        #  'type': 'line',
                                        #  'labels': ["training loglik (CSL)", "validation loglik (CSL)"],
                                        #  'title': "Loglikelihoods using CSL",
                                        #  'xlabel': "epoch",
                                        #  'ylabel': "loglik",
                                        #  },
                                        ])

    filter_display = Display(title="Receptive Fields",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 9,
                                       # 'disp_dim': (28, 28),
                                       'tile_shape': (3, 3),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/ssRBM/housing_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SemiSupervisedRBM(
        task='regression',
        num_hidden=10,
        num_visible=x_train.shape[1],
        batch_size=20,
        num_epochs=4,
        w_init=0.01,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='first_order',
        random_state=random_seed(),
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[filter_display, learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x, y)

    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))

    print("=========== Predicted by Semi-Supervised RBM ============")
    print("Train RMSE = {0:>1.4f}\tTest RMSE = {1:>1.4f}".format(
        -model.score(x_train, y_train),
        -model.score(x_test, y_test)))

    # fit a Support Vector Regressor
    s = SVR()
    s.fit(x_train, y_train)
    print("=========== Predicted by Support Vector Regressor ============")
    print("Train RMSE = {0:>1.4f}\tTest RMSE = {1:>1.4f}".format(
        np.sqrt(mean_squared_error(y_train, s.predict(x_train))),
        np.sqrt(mean_squared_error(y_test, s.predict(x_test)))))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_ssrbm_classification(show_figure=True, block_figure_on_end=True)
    # test_ssrbm_regression(show_figure=True, block_figure_on_end=True)
