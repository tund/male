from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pytest
import numpy as np

from male import model_dir
from male import random_seed
from male.datasets import demo
from male import TensorFlowModel
from male.callbacks import Display
from male.callbacks import ImageSaver
from male.callbacks import ModelCheckpoint
from male.models.deep_learning.generative import M2GAN


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_m2gan_mnist(block_figure_on_end=False):
    print("========== Test M2GAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
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
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = M2GAN(model_name="M2GAN_MNIST",
                  num_random_features=10,  # set to 1000 for a full run
                  gamma_init=0.01,
                  num_z=10,  # set to 100 for a full run
                  img_size=(28, 28, 1),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_m2gan_cifar10(block_figure_on_end=False):
    print("========== Test M2GAN on CIFAR10 data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
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
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = M2GAN(model_name="M2GAN_CIFAR10",
                  num_random_features=10,  # set 1000 for a full run
                  gamma_init=0.01,
                  num_z=10,  # set to 100 for a full run
                  img_size=(32, 32, 3),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=4,  # set to 500 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_m2gan_mnist(block_figure_on_end=True)
    # test_m2gan_cifar10(block_figure_on_end=True)
