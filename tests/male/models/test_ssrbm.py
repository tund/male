from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from male import model_dir
from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint
from male.models.deep_learning.rbm import SemiSupervisedRBM


def test_ssrbm_mnist():
    num_labeled_data = 1000

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    # remove some labels
    idx_train, idx_test = next(
        iter(StratifiedShuffleSplit(
            n_splits=1, test_size=num_labeled_data, random_state=6789).split(x_train, y_train)))
    y_train[idx_train] = 10 ** 8

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
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
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    filepath = os.path.join(model_dir(), "male/ssrbm/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SemiSupervisedRBM(
        num_hidden=6000,
        num_visible=784,
        batch_size=1000,
        num_epochs=1000,
        learning_rate=0.1,
        w_init=0.1,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        # inference_engine='gibbs',
        approx_method='first_order',
        random_state=6789,
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[filter_display, learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x, y)

    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))

    print("=========== Predicted by sRBM ============")
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


def test_ssrbm_mnist_regression():
    num_labeled_data = 1000

    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    # remove some labels
    idx_train, idx_test = next(
        iter(StratifiedShuffleSplit(
            n_splits=1, test_size=num_labeled_data, random_state=6789).split(x_train, y_train)))
    y_train[idx_train] = 10 ** 8

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
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
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    filepath = os.path.join(model_dir(), "male/ssrbm/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SemiSupervisedRBM(
        task='regression',
        num_hidden=1000,
        num_visible=784,
        batch_size=100,
        num_epochs=20,
        w_init=0.01,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        # inference_engine='gibbs',
        approx_method='first_order',
        # approx_method='second_order',
        random_state=6789,
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[filter_display, learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x, y)

    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))

    print("=========== Predicted by sRBM ============")
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


def test_ssrbm_diabetes_regression():
    '''TODO: remove some labels to test Semi-Supevised RBM
    '''
    from sklearn import datasets

    np.random.seed(random_seed())

    diabetes = datasets.load_diabetes()
    idx = np.random.permutation(diabetes.data.shape[0])
    x_train = diabetes.data[idx[:300]]
    y_train = diabetes.target[idx[:300]]
    x_test = diabetes.data[idx[300:]]
    y_test = diabetes.target[idx[300:]]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
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

    hidden_display = Display(title="Hidden Activations",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    filepath = os.path.join(model_dir(), "male/ssrbm/diabetes_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SemiSupervisedRBM(
        task='regression',
        num_hidden=20,
        num_visible=10,
        batch_size=100,
        num_epochs=1000,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        # inference_engine='gibbs',
        approx_method='second_order',
        random_state=6789,
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x, y)

    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))

    print("=========== Predicted by sRBM ============")
    print("Train accuracy = {0:>1.4f}\tTest accuracy = {1:>1.4f}".format(
        model.score(x_train, y_train),
        model.score(x_test, y_test)))


def test_ssrbm_load_to_continue_training():
    '''TODO: Fix the file name to test
    '''
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    model = SemiSupervisedRBM()
    model = model.load_model(os.path.join(model_dir(), "male/ssrbm/mnist_0030_0.423379.pkl"))
    model.fit(x, y)
    print("Test reconstruction error = %.4f" % model.get_reconstruction_error(x_test).mean())
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))


def test_ssrbm_mnist_gridsearch():
    '''TODO: remove some labels to test Semi-Supervised RBM
    '''
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import GridSearchCV

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    params = {'batch_size': [100, 1000],
              'learning_rate': [0.1, 0.01, 0.001],
              'weight_cost': [0.01, 0.001, 0.0001]}

    model = SemiSupervisedRBM(
        num_hidden=10,
        num_visible=784,
        batch_size=100,
        num_epochs=1000,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='second_order',
        random_state=6789,
        metrics=['loss'],
        callbacks=[early_stopping],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0] + [1] * x_test.shape[0])

    gs = GridSearchCV(model, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_ssrbm_mnist()
    # test_srbm_mnist_regression()
    # test_ssrbm_diabetes_regression()
    # test_srbm_mnist_gridsearch()
    # test_srbm_load_to_continue_training()
