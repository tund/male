from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn.datasets import load_svmlight_file

from male import HOME
from male.callbacks import Display
from male.models.distribution import Uniform
from male.models.deep_learning.generative import GAN


def test_gan_mnist():
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist"),
                                          n_features=784)
    x_train = x_train.toarray().astype(np.float32) / 255.0
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t"),
                                        n_features=784)
    x_test = x_test.toarray().astype(np.float32) / 255.0

    loss_display = Display(layout=(1, 1),
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    sample_display = Display(layout=(1, 1),
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

    model = GAN(num_x=784,
                num_discriminator_hiddens=(128,),
                discriminator_act_funcs=('relu',),
                discriminator_learning_rate=0.001,
                num_z=100,
                generator_distribution=Uniform(low=-1.0, high=1.0),
                num_generator_hiddens=(128,),
                generator_act_funcs=('relu',),
                generator_out_func='sigmoid',
                generator_learning_rate=0.001,
                batch_size=128,
                metrics=['d_loss', 'g_loss'],
                callbacks=[loss_display, sample_display],
                num_epochs=100,
                random_state=6789,
                verbose=1)
    model.fit(x_train)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_gan_mnist()
