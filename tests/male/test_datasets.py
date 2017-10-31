from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import pytest
import numpy as np
from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male.models.linear import GLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_fashion_mnist(show=False, block_figure_on_end=False):
    print("========== Test Fashion-MNIST dataset using GLM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_fashion_mnist()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_err', patience=2, verbose=1)
    filepath = os.path.join(model_dir(), "male/datasets/fashion_mnist_{epoch:04d}_{val_err:.6f}.pkl")
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
                             figsize=(3, 24),
                             freq=1,
                             show=show,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 1),
                                       },
                                      ])

    clf = GLM(model_name="GLM_softmax_cv",
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


if __name__ == '__main__':
    pytest.main([__file__])
    # test_fashion_mnist(show=True, block_figure_on_end=True)
