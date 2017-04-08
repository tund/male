from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male import Model
from male import data_dir
from male import model_dir
from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint
from male.models.deep_learning.rbm import SupervisedRBM
from male.models.deep_learning.rbm.rbm import INFERENCE_ENGINE
from male.models.deep_learning.rbm.srbm import APPROX_METHOD


def test_srbm_hidden_posterior_approximation(block_figure_on_end=False):
    print("========== Test hidden posterior approximation of Supervised RBM ==========")

    np.random.seed(random_seed())

    num_data = 10
    num_features = 4
    num_hidden = 3

    # Classification
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, 3, num_data)
    model = SupervisedRBM(
        task='classification',
        num_hidden=num_hidden,
        num_visible=num_features,
        batch_size=4,
        num_epochs=0,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='first_order',
        random_state=random_seed(),
        verbose=1)
    model.fit(x, y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['first_order']
    h_vi_1st = model._get_hidden_prob(x, y=y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['second_order']
    h_vi_2nd = model._get_hidden_prob(x, y=y)
    model.inference_engine = INFERENCE_ENGINE['gibbs']
    h_gibbs = model._get_hidden_prob(x, y=y)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)
    cax = [None] * 3
    cax[0] = axes[0].imshow(np.abs(h_vi_1st - h_vi_2nd),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[0].set_title("VI_1st vs VI_2nd", fontsize=24)
    cax[1] = axes[1].imshow(np.abs(h_vi_1st - h_gibbs),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[1].set_title("VI_1st vs Gibbs", fontsize=24)
    cax[2] = axes[2].imshow(np.abs(h_gibbs - h_vi_2nd),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[2].set_title("VI_2nd vs Gibbs", fontsize=24)
    for i in range(3):
        cb = fig.colorbar(cax[i], ax=axes[i])
        cb.ax.tick_params(labelsize=20)

    h_true = np.random.rand(num_data, num_hidden)
    v = model._get_visible_prob(h_true)
    y = model._predict_from_hidden(h_true)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['first_order']
    h_vi_1st = model._get_hidden_prob(v, y=y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['second_order']
    h_vi_2nd = model._get_hidden_prob(v, y=y)
    model.inference_engine = INFERENCE_ENGINE['gibbs']
    h_gibbs = model._get_hidden_prob(v, y=y)
    fig, axes = plt.subplots(1, 3)
    cax = [None] * 3
    cax[0] = axes[0].imshow(np.abs(h_vi_1st - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[0].set_title("VI_1st vs True", fontsize=24)
    cax[1] = axes[1].imshow(np.abs(h_vi_2nd - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[1].set_title("VI_2nd vs True", fontsize=24)
    cax[2] = axes[2].imshow(np.abs(h_gibbs - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[2].set_title("Gibbs vs True", fontsize=24)
    for i in range(3):
        cb = fig.colorbar(cax[i], ax=axes[i])
        cb.ax.tick_params(labelsize=20)

    # Regression
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = SupervisedRBM(
        task='regression',
        num_hidden=num_hidden,
        num_visible=num_features,
        batch_size=4,
        num_epochs=0,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='first_order',
        random_state=random_seed(),
        verbose=1)
    model.fit(x, y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['first_order']
    h_vi_1st = model._get_hidden_prob(x, y=y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['second_order']
    h_vi_2nd = model._get_hidden_prob(x, y=y)
    model.inference_engine = INFERENCE_ENGINE['gibbs']
    h_gibbs = model._get_hidden_prob(x, y=y)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3)
    cax = [None] * 3
    cax[0] = axes[0].imshow(np.abs(h_vi_1st - h_vi_2nd),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[0].set_title("VI_1st vs VI_2nd", fontsize=24)
    cax[1] = axes[1].imshow(np.abs(h_vi_1st - h_gibbs),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[1].set_title("VI_1st vs Gibbs", fontsize=24)
    cax[2] = axes[2].imshow(np.abs(h_gibbs - h_vi_2nd),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[2].set_title("VI_2nd vs Gibbs", fontsize=24)
    for i in range(3):
        cb = fig.colorbar(cax[i], ax=axes[i])
        cb.ax.tick_params(labelsize=20)

    h_true = np.random.rand(num_data, num_hidden)
    v = model._get_visible_prob(h_true)
    y = model._predict_from_hidden(h_true).ravel()
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['first_order']
    h_vi_1st = model._get_hidden_prob(v, y=y)
    model.inference_engine = INFERENCE_ENGINE['variational_inference']
    model.approx_method = APPROX_METHOD['second_order']
    h_vi_2nd = model._get_hidden_prob(v, y=y)
    model.inference_engine = INFERENCE_ENGINE['gibbs']
    h_gibbs = model._get_hidden_prob(v, y=y)
    fig, axes = plt.subplots(1, 3)
    cax = [None] * 3
    cax[0] = axes[0].imshow(np.abs(h_vi_1st - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[0].set_title("VI_1st vs True", fontsize=24)
    cax[1] = axes[1].imshow(np.abs(h_vi_2nd - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[1].set_title("VI_2nd vs True", fontsize=24)
    cax[2] = axes[2].imshow(np.abs(h_gibbs - h_true),
                            aspect='auto', interpolation='None', cmap='jet')
    axes[2].set_title("Gibbs vs True", fontsize=24)
    for i in range(3):
        cb = fig.colorbar(cax[i], ax=axes[i])
        cb.ax.tick_params(labelsize=20)

    plt.show(block=block_figure_on_end)


def test_srbm_classification(block_figure_on_end=False):
    print("========== Test Supervised RBM for Classification ==========")

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
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
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/sRBM/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SupervisedRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='second_order',
        metrics=['recon_err', 'loss', 'err'],
        callbacks=[filter_display, learning_display,
                   hidden_display, early_stopping, checkpoint],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
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


def test_srbm_regression(block_figure_on_end=False):
    print("========== Test Supervised RBM for Regression ==========")

    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(3, 1),
                               freq=1,
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
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:100],
                                       },
                                      ])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/sRBM/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    model = SupervisedRBM(
        task='regression',
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        w_init=0.01,
        learning_rate=0.01,
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
    print("Test loss = %.4f" % model.get_loss(x_test, y_test))

    print("=========== Predicted by sRBM ============")
    print("Train MSE = {0:>1.4f}\tTest MSE = {1:>1.4f}".format(
        -model.score(x_train, y_train),
        -model.score(x_test, y_test)))

    # fit a Linear Regressor
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("=========== Predicted by Linear Regressor ============")
    print("Train MSE = {0:>1.4f}\tTest MSE = {1:>1.4f}".format(
        mean_squared_error(y_train, lr.predict(x_train)),
        mean_squared_error(y_test, lr.predict(x_test))))


def test_srbm_load_to_continue_training():
    print("========== Test load to continue training Supervised RBM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()

    model = SupervisedRBM(
        num_hidden=5,
        num_visible=x_train.shape[1],
        batch_size=10,
        num_epochs=2,
        learning_rate=0.1,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='second_order',
        metrics=['recon_err', 'loss', 'err'],
        random_state=random_seed(),
        verbose=1)

    model.fit(x_train, y_train)

    print("After training for {0:d} epochs".format(model.num_epochs))
    train_err = 1.0 - model.score(x_train, y_train)
    test_err = 1.0 - model.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    save_file_path = model.save()
    model1 = Model.load_model(save_file_path)
    print("After save and load:")
    train_err1 = 1.0 - model1.score(x_train, y_train)
    test_err1 = 1.0 - model1.score(x_test, y_test)
    print("Training error = %.4f" % train_err1)
    print("Testing error = %.4f" % test_err1)
    assert abs(train_err - train_err1) < 1e-6
    assert abs(test_err - test_err1) < 1e-6

    model.num_epochs = 4
    model.fit(x_train, y_train)
    print("Set number of epoch to {0:d}, then continue training...".format(model.num_epochs))
    train_err = 1.0 - model.score(x_train, y_train)
    test_err = 1.0 - model.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_srbm_gridsearch():
    print("========== Tune parameters for Supervised RBM ==========")

    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import GridSearchCV

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    params = {'batch_size': [64, 100],
              'learning_rate': [0.1, 0.01],
              'weight_cost': [0.001, 0.0001]}

    model = SupervisedRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        learning_rate=0.01,
        momentum_method='sudden',
        weight_cost=0.0,
        inference_engine='variational_inference',
        approx_method='first_order',
        metrics=['loss'],
        callbacks=[early_stopping],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=0)

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0] + [1] * x_test.shape[0])

    gs = GridSearchCV(model, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1.0 - gs.best_score_, gs.best_params_))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_srbm_hidden_posterior_approximation(block_figure_on_end=True)
    # test_srbm_classification(block_figure_on_end=True)
    # test_srbm_regression(block_figure_on_end=True)
    # test_srbm_load_to_continue_training()
    # test_srbm_gridsearch()
