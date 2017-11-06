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
from male.models.deep_learning.generative import DCGAN


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN on MNIST data ==========")

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

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  z_prior=Uniform1D(low=-1.0, high=1.0),
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

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  z_prior=Gaussian1D(mu=0.0, sigma=1.0),
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
def test_dcgan_fashion_mnist(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN on Fashion-MNIST data ==========")

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

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  z_prior=Uniform1D(low=-1.0, high=1.0),
                  img_size=(28, 28, 1),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 64 for a full run
                  num_dis_feature_maps=4,  # set to 64 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  z_prior=Gaussian1D(mu=0.0, sigma=1.0),
                  img_size=(28, 28, 1),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 64 for a full run
                  num_dis_feature_maps=4,  # set to 64 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_save_and_load(show_figure=False, block_figure_on_end=False):
    print("========== Test Save and Load functions of DCGAN on MNIST data ==========")

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

    model = DCGAN(model_name="DCGAN_MNIST_SaveLoad",
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

    save_file_path = model.save()

    model1 = TensorFlowModel.load_model(save_file_path)
    model1.num_epochs = 10
    model1.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN on CIFAR10 data ==========")

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
def test_dcgan_cifar10_inception_score(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN with Inception Score on CIFAR10 data ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    filepath = os.path.join(model_dir(), "male/DCGAN/cifar10/checkpoints/"
                                         "{epoch:04d}_{inception_score:.6f}.ckpt")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='max',
                                 monitor='inception_score',
                                 verbose=1,
                                 save_best_only=True)
    loss_display = Display(layout=(1, 1),
                           dpi='auto',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           filepath=[os.path.join(model_dir(), "male/DCGAN/cifar10/"
                                                               "loss/loss_{epoch:04d}.png"),
                                     os.path.join(model_dir(), "male/DCGAN/cifar10/"
                                                               "loss/loss_{epoch:04d}.pdf")],
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
                                      show=show_figure,
                                      block_on_end=block_figure_on_end,
                                      filepath=[os.path.join(model_dir(),
                                                             "male/DCGAN/cifar10/inception_score/"
                                                             "inception_score_{epoch:04d}.png"),
                                                os.path.join(model_dir(),
                                                             "male/DCGAN/cifar10/inception_score/"
                                                             "inception_score_{epoch:04d}.pdf")],
                                      monitor=[{'metrics': ['inception_score'],
                                                'type': 'line',
                                                'labels': ["Inception Score"],
                                                'title': "Scores",
                                                'xlabel': "epoch",
                                                'ylabel': "score",
                                                },
                                               ],
                                      )
    sample_display = Display(layout=(1, 1),
                             dpi='auto',
                             figsize=(10, 10),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             filepath=os.path.join(model_dir(),
                                                   "male/DCGAN/cifar10/samples/"
                                                   "samples_{epoch:04d}.png"),
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
                  callbacks=[loss_display, inception_score_display, sample_display, checkpoint],
                  num_epochs=4,  # set to 100 for a full run
                  inception_score_freq=1,
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_image_saver():
    print("========== Test DCGAN with Image Saver ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    imgsaver = ImageSaver(freq=1,
                          filepath=os.path.join(model_dir(), "male/DCGAN/imagesaver/"
                                                             "mnist/mnist_{epoch:04d}.png"),
                          monitor={'metrics': 'x_samples',
                                   'num_samples': 100,
                                   'tile_shape': (10, 10),
                                   })

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  img_size=(28, 28, 1),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[imgsaver],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    imgsaver = ImageSaver(freq=1,
                          filepath=os.path.join(model_dir(), "male/DCGAN/imagesaver/"
                                                             "cifar10/cifar10_{epoch:04d}.png"),
                          monitor={'metrics': 'x_samples',
                                   'num_samples': 100,
                                   'tile_shape': (10, 10),
                                   })

    model = DCGAN(model_name="DCGAN_CIFAR10",
                  num_z=10,  # set to 100 for a full run
                  img_size=(32, 32, 3),
                  batch_size=32,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[imgsaver],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_dcgan_mnist(show_figure=True, block_figure_on_end=True)
    # test_dcgan_fashion_mnist(show_figure=True, block_figure_on_end=True)
    # test_dcgan_save_and_load(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10_inception_score(show_figure=True, block_figure_on_end=True)
    # test_dcgan_image_saver()
