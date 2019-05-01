from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male import PyTorchModel
from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male.models.linear import PyTorchGLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_pytorch_glm_logit():
    print('========== Test PytorchGLM for binary classification ==========')

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_pima()
    print('Number of training samples = {}'.format(x_train.shape[0]))
    print('Number of testing samples = {}'.format(x_test.shape[0]))

    clf = PyTorchGLM(model_name='PytorchGLM_logit',
                     l1_penalty=0.0,
                     l2_penalty=0.0,
                     optimizer='sgd',
                     random_state=random_seed())

    print('Use {} optimizer'.format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print('Training error = %.4f' % train_err)
    print('Testing error = %.4f' % test_err)


def test_pytorch_glm_softmax():
    print("========== Test PytorchGLM for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = PyTorchGLM(model_name="PytorchGLM_softmax",
                     link='softmax',
                     loss='softmax',
                     optimizer='sgd',
                     random_state=random_seed())

    print("Use {} optimizer".format(clf.optimizer))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_pytorch_glm_regression():
    print("========== Test PytorchGLM for regression ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_housing()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = PyTorchGLM(model_name="PytorchGLM_regression",
                     task='regression',
                     link='linear',  # link function
                     loss='quadratic',  # loss function
                     l2_penalty=0.0,  # ridge regularization
                     l1_penalty=0.0,  # Lasso regularization
                     l1_smooth=1E-5,  # smoothing for Lasso regularization
                     l1_method='pseudo_huber',  # approximation method for L1-norm
                     random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = - clf.score(x_train, y_train)
    test_err = - clf.score(x_test, y_test)
    print("Training MSE = %.4f" % train_err)
    print("Testing MSE = %.4f" % test_err)


def test_pytorch_glm_cv(show=False, block_figure_on_end=False):
    print("========== Test cross-validation for PytorchGLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.uint8)
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/PyTorchGLM/iris_{epoch:04d}_{val_err:.6f}.pkl")
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

    clf = PyTorchGLM(model_name="PyTorchGLM_softmax_cv",
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


def test_pytorch_glm_save_load(show=False, block_figure_on_end=False):
    print("========== Test Save and Load functions for PyTorchGLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.uint8)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.uint8)
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=5, verbose=1)
    filepath = os.path.join(model_dir(), "male/PyTorchGLM/iris_{epoch:04d}_{val_err:.6f}.pkl")
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

    clf = PyTorchGLM(model_name="PyTorchGLM_softmax_cv",
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

    save_file_path = os.path.join(model_dir(), "male/PyTorchGLM/saved_model.pkl")
    clf.save(file_path=save_file_path)
    clf1 = PyTorchModel.load_model(save_file_path)
    clf1.num_epochs = 10
    clf1.fit(x, y)

    train_err = 1.0 - clf1.score(x_train, y_train)
    test_err = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_pytorch_glm_logit()
    # test_pytorch_glm_softmax()
    # test_pytorch_glm_regression()
    # test_pytorch_glm_cv(show=True, block_figure_on_end=True)
    # test_pytorch_glm_save_load(show=True, block_figure_on_end=True)
