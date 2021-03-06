from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pytest
import numpy as np

from male.configs import model_dir
from male.configs import random_seed
from male.datasets import demo
from male import TensorFlowModel
from male.callbacks import Display
from male.callbacks import ImageSaver
from male.callbacks import ModelCheckpoint
from male.models.distributions import Uniform1D
from male.models.deep_learning.generative import CGAN
from male.models.deep_learning.generative import CGANv1


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_cgan_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test CGAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = CGAN(model_name="CGAN_MNIST",
                 num_z=10,  # set to 100 for a full run
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 img_size=(28, 28, 1),
                 batch_size=64,  # set to 64 for a full run
                 num_conv_layers=3,  # set to 3 for a full run
                 num_gen_feature_maps=2,  # set to 32 for a full run
                 num_dis_feature_maps=2,  # set to 32 for a full run
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=1,  # set to 100 for a full run
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train, y_train)

    # <editor-fold desc="CGAN with Gaussian noise">
    # model = CGAN(model_name="CGAN_MNIST",
    #              num_z=10,  # set to 100 for a full run
    #              z_prior=Gaussian1D(mu=0.0, sigma=1.0),
    #              img_size=(28, 28, 1),
    #              batch_size=32,  # set to 64 for a full run
    #              num_conv_layers=3,  # set to 3 for a full run
    #              num_gen_feature_maps=4,  # set to 32 for a full run
    #              num_dis_feature_maps=4,  # set to 32 for a full run
    #              metrics=['d_loss', 'g_loss'],
    #              callbacks=[loss_display, sample_display],
    #              num_epochs=4,  # set to 100 for a full run
    #              random_state=random_seed(),
    #              verbose=1)
    #
    # model.fit(x_train, y_train)
    # </editor-fold>


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_cganv1_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test CGAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = CGANv1(model_name="CGANv1_MNIST",
                   num_z=10,  # set to 100 for a full run
                   z_prior=Uniform1D(low=-1.0, high=1.0),
                   img_size=(28, 28, 1),
                   batch_size=64,  # set to 64 for a full run
                   num_conv_layers=3,  # set to 3 for a full run
                   num_gen_feature_maps=2,  # set to 32 for a full run
                   num_dis_feature_maps=2,  # set to 32 for a full run
                   metrics=['d_loss', 'g_loss'],
                   callbacks=[loss_display, sample_display],
                   num_epochs=1,  # set to 100 for a full run
                   random_state=random_seed(),
                   verbose=1)

    model.fit(x_train, y_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_cgan_fashion_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test CGAN on Fashion-MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_fashion_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "CGAN: Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "CGAN: Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = CGAN(model_name="CGAN_Fashion-MNIST",
                 num_z=10,  # set to 100 for a full run
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 img_size=(28, 28, 1),
                 batch_size=64,  # set to 64 for a full run
                 num_conv_layers=3,  # set to 3 for a full run
                 num_gen_feature_maps=2,  # set to 64 for a full run
                 num_dis_feature_maps=2,  # set to 64 for a full run
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=1,  # set to 100 for a full run
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train, y_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_cgan_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test CGAN on CIFAR10 data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = CGAN(model_name="CGAN_CIFAR10",
                 num_z=10,  # set to 100 for a full run
                 img_size=(32, 32, 3),
                 batch_size=64,  # set to 64 for a full run
                 num_conv_layers=3,  # set to 3 for a full run
                 num_gen_feature_maps=2,  # set to 32 for a full run
                 num_dis_feature_maps=2,  # set to 32 for a full run
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=1,  # set to 100 for a full run
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train, y_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_cganv1_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test CGANv1 on CIFAR10 data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "CGANv1 - Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "CGANv1 - Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = CGANv1(model_name="CGANv1_CIFAR10",
                   num_z=10,  # set to 100 for a full run
                   img_size=(32, 32, 3),
                   batch_size=64,  # set to 64 for a full run
                   num_conv_layers=3,  # set to 3 for a full run
                   num_gen_feature_maps=2,  # set to 32 for a full run
                   num_dis_feature_maps=2,  # set to 32 for a full run
                   metrics=['d_loss', 'g_loss'],
                   callbacks=[loss_display, sample_display],
                   num_epochs=1,  # set to 100 for a full run
                   random_state=random_seed(),
                   verbose=1)

    model.fit(x_train, y_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_cgan_mnist(show_figure=True, block_figure_on_end=True)
    # test_cganv1_mnist(show_figure=True, block_figure_on_end=True)
    # test_cgan_fashion_mnist(show_figure=True, block_figure_on_end=True)
    # test_cgan_cifar10(show_figure=True, block_figure_on_end=True)
    # test_cganv1_cifar10(show_figure=True, block_figure_on_end=True)
