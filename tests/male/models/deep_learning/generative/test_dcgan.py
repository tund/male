from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.generative import DCGAN


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_mnist(block_figure_on_end=False):
    print("========== Test DCGAN on MNIST data ==========")

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

    model = DCGAN(model_name="DCGAN_MNIST",
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
def test_dcgan_cifar10(block_figure_on_end=False):
    print("========== Test DCGAN on CIFAR10 data ==========")

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

    model = DCGAN(model_name="DCGAN_CIFAR10",
                  num_z=10,  # set to 100 for a full run
                  img_size=(32, 32, 3),
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
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_dcgan_cifar10_inception_score(block_figure_on_end=False):
    print("========== Test DCGAN with Inception Score on CIFAR10 data ==========")

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
    inception_score_display = Display(layout=(1, 1),
                                      dpi='auto',
                                      block_on_end=block_figure_on_end,
                                      monitor=[{'metrics': ['inception_score'],
                                                'type': 'line',
                                                'labels': ["Inception Score"],
                                                'title': "Scores",
                                                'xlabel': "epoch",
                                                'ylabel': "score",
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

    model = DCGAN(model_name="DCGAN_CIFAR10",
                  num_z=10,  # set to 100 for a full run
                  img_size=(32, 32, 3),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss', 'inception_score'],
                  callbacks=[loss_display, inception_score_display, sample_display],
                  num_epochs=100,  # set to 100 for a full run
                  inception_score_freq=1,
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_dcgan_mnist(block_figure_on_end=True)
    # test_dcgan_cifar10(block_figure_on_end=True)
    # test_dcgan_cifar10_inception_score(block_figure_on_end=True)
