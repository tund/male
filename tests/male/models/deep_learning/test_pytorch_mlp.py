from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male.datasets import demo
from male.configs import model_dir
from male.configs import random_seed
from male.callbacks import Display
from male.models.deep_learning import PyTorchMLP


def test_pytorch_mlp_v1(show=False, block_figure_on_end=False):
    print("========== Test PytorchMLPv1 ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train].astype(np.float32)
    y_train = y_train[idx_train].astype(np.uint8)
    print("Number of training samples = {}".format(x_train.shape[0]))

    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test].astype(np.float32)
    y_test = y_test[idx_test].astype(np.uint8)
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    err_display = Display(title="Error curves",
                          dpi='auto',
                          layout=(1, 1),
                          freq=1,
                          show=show,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['err', 'val_err'],
                                    'type': 'line',
                                    'title': "Learning errors",
                                    'xlabel': "epoch",
                                    'ylabel': "error",
                                    }])
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
                           show=show,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(model_dir(), "male/PyTorchMLP/"
                                                               "loss/loss_{epoch:04d}.png"),
                                     os.path.join(model_dir(), "male/PyTorchMLP/"
                                                               "loss/loss_{epoch:04d}.pdf")],
                           monitor=[{'metrics': ['loss', 'val_loss'],
                                     'type': 'line',
                                     'labels': ["training loss", "validation loss"],
                                     'title': "Learning losses",
                                     'xlabel': "epoch",
                                     'xlabel_params': {'fontsize': 50},
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
                             filepath=os.path.join(model_dir(), "male/PyTorchMLP/"
                                                                "weights/weights_{epoch:04d}.png"),
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    clf = PyTorchMLP(model_name='PyTorchMLP',
                     arch='MLPv1',
                     num_epochs=4,
                     batch_size=100,
                     metrics=['loss', 'err'],
                     callbacks=[loss_display, err_display, weight_display],
                     cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
                     random_state=random_seed(),
                     verbose=1)

    clf.fit(x, y)
    print("Training error = %.4f" % (1.0 - clf.score(x_train, y_train)))
    print("Testing error = %.4f" % (1.0 - clf.score(x_test, y_test)))


def test_pytorch_mlp_v2(show=False, block_figure_on_end=False):
    print("========== Test PytorchMLPv2 ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train].astype(np.float32)
    y_train = y_train[idx_train].astype(np.uint8)
    print("Number of training samples = {}".format(x_train.shape[0]))

    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test].astype(np.float32)
    y_test = y_test[idx_test].astype(np.uint8)
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    err_display = Display(title="Error curves",
                          dpi='auto',
                          layout=(1, 1),
                          freq=1,
                          show=show,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['err', 'val_err'],
                                    'type': 'line',
                                    'title': "Learning errors",
                                    'xlabel': "epoch",
                                    'ylabel': "error",
                                    }])
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=1,
                           show=show,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(model_dir(), "male/PyTorchMLP/"
                                                               "loss/loss_{epoch:04d}.png"),
                                     os.path.join(model_dir(), "male/PyTorchMLP/"
                                                               "loss/loss_{epoch:04d}.pdf")],
                           monitor=[{'metrics': ['loss', 'val_loss'],
                                     'type': 'line',
                                     'labels': ["training loss", "validation loss"],
                                     'title': "Learning losses",
                                     'xlabel': "epoch",
                                     'xlabel_params': {'fontsize': 50},
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
                             filepath=os.path.join(model_dir(), "male/PyTorchMLP/"
                                                                "weights/weights_{epoch:04d}.png"),
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    clf = PyTorchMLP(model_name='PyTorchMLP',
                     arch='MLPv2',
                     num_epochs=4,
                     batch_size=100,
                     metrics=['loss', 'err'],
                     callbacks=[loss_display, err_display, weight_display],
                     cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
                     random_state=random_seed(),
                     verbose=1)

    clf.fit(x, y)
    print("Training error = %.4f" % (1.0 - clf.score(x_train, y_train)))
    print("Testing error = %.4f" % (1.0 - clf.score(x_test, y_test)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_pytorch_mlp_v1(show=True, block_figure_on_end=True)
    # test_pytorch_mlp_v2(show=True, block_figure_on_end=True)
