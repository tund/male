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
from male.models.distribution import Uniform1D
from male.models.distribution import Gaussian1D
from male.models.deep_learning.generative import WGAN


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_wgan_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test WGAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/WGAN/MNIST")
    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(root_dir, "loss/loss_{epoch:04d}.png"),
                                     os.path.join(root_dir, "loss/loss_{epoch:04d}.pdf")],
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
                             filepath=os.path.join(root_dir, "samples/samples_{epoch:04d}.png"),
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = WGAN(model_name="WGAN_MNIST",
                 num_z=10,  # set to 100 for a full run
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 img_size=(28, 28, 1),
                 batch_size=16,  # set to 64 for a full run
                 num_conv_layers=3,  # set to 3 for a full run
                 num_gen_feature_maps=4,  # set to 32 for a full run
                 num_dis_feature_maps=4,  # set to 32 for a full run
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=4,  # set to 100 for a full run
                 log_path=os.path.join(root_dir, "logs"),
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_wgan_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test WGAN on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/WGAN/CIFAR10")
    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(root_dir, "loss/loss_{epoch:04d}.png"),
                                     os.path.join(root_dir, "loss/loss_{epoch:04d}.pdf")],
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
                             filepath=os.path.join(root_dir, "samples/samples_{epoch:04d}.png"),
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = WGAN(model_name="WGAN_CIFAR10",
                 num_z=10,  # set to 100 for a full run
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 img_size=(32, 32, 3),
                 batch_size=16,  # set to 64 for a full run
                 num_conv_layers=3,  # set to 3 for a full run
                 num_gen_feature_maps=4,  # set to 32 for a full run
                 num_dis_feature_maps=4,  # set to 32 for a full run
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=4,  # set to 100 for a full run
                 log_path=os.path.join(root_dir, "logs"),
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_wgan_save_and_load(show_figure=False, block_figure_on_end=False):
    print("========== Test Save and Load functions of WGAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    num_data = 128
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/WGAN/MNIST")
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

    model = WGAN(model_name="WGAN_MNIST_SaveLoad",
                 num_z=8,
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 img_size=(28, 28, 1),
                 batch_size=16,
                 num_conv_layers=3,
                 num_gen_feature_maps=4,
                 num_dis_feature_maps=4,
                 metrics=['d_loss', 'g_loss'],
                 callbacks=[loss_display, sample_display],
                 num_epochs=2,
                 log_path=os.path.join(root_dir, "logs"),
                 random_state=random_seed(),
                 verbose=1)

    model.fit(x_train)

    print("Saving model...")
    save_file_path = model.save(os.path.join(root_dir, "checkpoints/ckpt"))
    print("Reloading model...")
    model1 = TensorFlowModel.load_model(save_file_path)
    model1.num_epochs = 4
    model1.fit(x_train)
    print("Done!")


if __name__ == '__main__':
    pytest.main([__file__])
    # test_wgan_mnist(show_figure=True, block_figure_on_end=True)
    # test_wgan_cifar10(show_figure=True, block_figure_on_end=True)
    # test_wgan_save_and_load(show_figure=False, block_figure_on_end=False)
