from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.distribution import Uniform
from male.models.deep_learning.generative import GAN


def test_gan_mnist():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    loss_display = Display(layout=(1, 1),
                           dpi='auto',
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
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    # <editor-fold desc="Working example">
    model = GAN(num_x=784,
                num_discriminator_hiddens=(128,),
                discriminator_batchnorm=False,
                discriminator_act_funcs=('lrelu',),
                discriminator_learning_rate=0.001,
                num_z=100,
                generator_distribution=Uniform(low=(-1.0,) * 100, high=(1.0,) * 100),
                generator_batchnorm=False,
                num_generator_hiddens=(128,),
                generator_act_funcs=('lrelu',),
                generator_out_func='sigmoid',
                generator_learning_rate=0.001,
                batch_size=128,
                metrics=['d_loss', 'g_loss'],
                callbacks=[loss_display, sample_display],
                num_epochs=100,
                random_state=6789,
                verbose=1)
    # </editor-fold>

    # <editor-fold desc="Testing example">
    # model = GAN(num_x=784,
    #             num_discriminator_hiddens=(1024, 512, 256),
    #             discriminator_batchnorm=False,
    #             discriminator_act_funcs=('lrelu', 'lrelu', 'lrelu'),
    #             discriminator_learning_rate=0.001,
    #             num_z=100,
    #             # generator_distribution=Gaussian(mu=0.0, sigma=1.0),
    #             generator_distribution=Uniform(low=0.0, high=1.0),
    #             generator_batchnorm=False,
    #             num_generator_hiddens=(256, 512, 1024),
    #             generator_act_funcs=('lrelu', 'lrelu', 'lrelu'),
    #             generator_out_func='sigmoid',
    #             generator_learning_rate=0.0001,
    #             batch_size=128,
    #             metrics=['d_loss', 'g_loss'],
    #             callbacks=[loss_display, sample_display],
    #             num_epochs=100,
    #             random_state=6789,
    #             verbose=1)
    # </editor-fold>

    model.fit(x_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_gan_mnist()
