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
from male.models.distributions import Uniform
from male.models.deep_learning.generative import GAN


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_gan_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test GAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

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
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    # <editor-fold desc="Working example">
    # model = GAN(num_x=784,
    #             num_discriminator_hiddens=(128,),
    #             discriminator_batchnorm=False,
    #             discriminator_act_funcs=('lrelu',),
    #             discriminator_learning_rate=0.001,
    #             num_z=100,
    #             generator_distribution=Uniform(low=(-1.0,) * 100, high=(1.0,) * 100),
    #             generator_batchnorm=False,
    #             num_generator_hiddens=(128,),
    #             generator_act_funcs=('lrelu',),
    #             generator_out_func='sigmoid',
    #             generator_learning_rate=0.001,
    #             batch_size=32,
    #             metrics=['d_loss', 'g_loss'],
    #             callbacks=[loss_display, sample_display],
    #             num_epochs=100,
    #             random_state=random_seed(),
    #             verbose=1)
    # </editor-fold>

    # <editor-fold desc="Testing example">
    NUM_Z = 10  # set to 100 for a full run
    model = GAN(num_x=784,
                num_discriminator_hiddens=(16,),  # set to 128 for a full run
                discriminator_batchnorm=False,
                discriminator_act_funcs=('lrelu',),
                discriminator_dropouts=(0.99,),
                discriminator_learning_rate=0.001,
                num_z=NUM_Z,
                generator_distribution=Uniform(low=(-1.0,) * NUM_Z, high=(1.0,) * NUM_Z),
                generator_batchnorm=False,
                num_generator_hiddens=(16, 16),  # set to (128, 128) for a full run
                generator_act_funcs=('lrelu', 'lrelu'),
                generator_out_func='sigmoid',
                generator_learning_rate=0.001,
                batch_size=32,
                metrics=['d_loss', 'g_loss'],
                callbacks=[loss_display, sample_display],
                num_epochs=4,  # set to 100 for a full run
                random_state=random_seed(),
                verbose=1)
    # </editor-fold>

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_gan_fashion_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test GAN on Fashion-MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_fashion_mnist()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

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
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    # <editor-fold desc="Working example">
    # model = GAN(num_x=784,
    #             num_discriminator_hiddens=(128,),
    #             discriminator_batchnorm=False,
    #             discriminator_act_funcs=('lrelu',),
    #             discriminator_learning_rate=0.001,
    #             num_z=100,
    #             generator_distribution=Uniform(low=(-1.0,) * 100, high=(1.0,) * 100),
    #             generator_batchnorm=False,
    #             num_generator_hiddens=(128,),
    #             generator_act_funcs=('lrelu',),
    #             generator_out_func='sigmoid',
    #             generator_learning_rate=0.001,
    #             batch_size=32,
    #             metrics=['d_loss', 'g_loss'],
    #             callbacks=[loss_display, sample_display],
    #             num_epochs=100,
    #             random_state=random_seed(),
    #             verbose=1)
    # </editor-fold>

    # <editor-fold desc="Testing example">
    NUM_Z = 10  # set to 100 for a full run
    model = GAN(num_x=784,
                num_discriminator_hiddens=(16,),  # set to 128 for a full run
                discriminator_batchnorm=False,
                discriminator_act_funcs=('lrelu',),
                discriminator_dropouts=(0.99,),
                discriminator_learning_rate=0.001,
                num_z=NUM_Z,
                generator_distribution=Uniform(low=(-1.0,) * NUM_Z, high=(1.0,) * NUM_Z),
                generator_batchnorm=False,
                num_generator_hiddens=(16, 16),  # set to (128, 128) for a full run
                generator_act_funcs=('lrelu', 'lrelu'),
                generator_out_func='sigmoid',
                generator_learning_rate=0.001,
                batch_size=32,
                metrics=['d_loss', 'g_loss'],
                callbacks=[loss_display, sample_display],
                num_epochs=4,  # set to 100 for a full run
                random_state=random_seed(),
                verbose=1)
    # </editor-fold>

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_gan_save_and_load(show_figure=False, block_figure_on_end=False):
    print("========== Test Save and Load functions of GAN on MNIST data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/GAN/MNIST")
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

    model = GAN(model_name="GAN_MNIST_SaveLoad",
                num_x=784,
                num_discriminator_hiddens=(16,),
                discriminator_batchnorm=False,
                discriminator_act_funcs=('lrelu',),
                discriminator_learning_rate=0.001,
                num_z=8,
                generator_distribution=Uniform(low=(-1.0,) * 8, high=(1.0,) * 8),
                generator_batchnorm=False,
                num_generator_hiddens=(16,),
                generator_act_funcs=('lrelu',),
                generator_out_func='sigmoid',
                generator_learning_rate=0.001,
                batch_size=32,
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
    # test_gan_mnist(show_figure=True, block_figure_on_end=True)
    # test_gan_fashion_mnist(show_figure=True, block_figure_on_end=True)
    # test_gan_save_and_load(show_figure=False, block_figure_on_end=False)
