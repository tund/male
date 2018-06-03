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
from male.metrics import FID
from male.metrics import InceptionScore
from male.metrics import InceptionMetricList
from male.models.distributions import Uniform1D
from male.models.distributions import Gaussian1D
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

    model = DCGAN(model_name="DCGAN_MNIST_z_uniform",
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
                  # summary_freq=1,  # uncomment this for a full run
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)

    model = DCGAN(model_name="DCGAN_MNIST_z_Gaussian",
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
                  # summary_freq=1,  # uncomment this for a full run
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
                  batch_size=16,  # set to 64 for a full run
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

    root_dir = os.path.join(model_dir(), "male/DCGAN/MNIST")
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
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=2,  # set to 100 for a full run
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


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/cifar10")
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
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, sample_display],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  log_path=os.path.join(root_dir, "logs"),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_dcgan_cifar10_inception_score(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN with Inception Score on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/cifar10")
    filepath = os.path.join(root_dir, "checkpoints/{epoch:04d}_{inception_score:.6f}.ckpt")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='max',
                                 monitor='inception_score',
                                 verbose=1,
                                 save_best_only=True)
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
    inception_score_display = Display(layout=(1, 1),
                                      dpi='auto',
                                      show=show_figure,
                                      block_on_end=block_figure_on_end,
                                      filepath=[os.path.join(root_dir,
                                                             "inception_score/"
                                                             "inception_score_{epoch:04d}.png"),
                                                os.path.join(root_dir,
                                                             "inception_score/"
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
                             filepath=os.path.join(root_dir, "samples/"
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
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss', 'inception_score'],
                  callbacks=[loss_display, inception_score_display, sample_display, checkpoint],
                  num_epochs=4,  # set to 100 for a full run
                  inception_metrics=InceptionScore(),
                  inception_metrics_freq=1,
                  # summary_freq=1,  # uncomment this for a full run
                  log_path=os.path.join(root_dir, "logs"),
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_dcgan_cifar10_fid(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN with FID on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/cifar10")
    checkpoints = ModelCheckpoint(
        os.path.join(root_dir, "checkpoints/{epoch:04d}_{FID:.6f}.ckpt"),
        mode='min',
        monitor='FID',
        verbose=1,
        save_best_only=True)
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
    fid_display = Display(layout=(1, 1),
                          dpi='auto',
                          show=show_figure,
                          block_on_end=block_figure_on_end,
                          filepath=[os.path.join(root_dir, "FID/"
                                                           "FID_{epoch:04d}.png"),
                                    os.path.join(root_dir, "FID/"
                                                           "FID_{epoch:04d}.pdf")],
                          monitor=[{'metrics': ['FID'],
                                    'type': 'line',
                                    'labels': ["FID"],
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
                             filepath=os.path.join(root_dir, "samples/"
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
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss', 'FID', 'FID_100points'],
                  callbacks=[loss_display, fid_display,
                             sample_display, checkpoints],
                  num_epochs=4,  # set to 100 for a full run
                  inception_metrics=[FID(data="cifar10"),
                                     FID(name="FID_100points", data=x_train[:100])],
                  inception_metrics_freq=1,
                  # summary_freq=1,  # uncomment this for a full run
                  log_path=os.path.join(root_dir, "logs"),
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_dcgan_cifar10_inception_metric(show_figure=False, block_figure_on_end=False):
    print("========== Test DCGAN with Inception Score and FID on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/cifar10")
    checkpoints_is = ModelCheckpoint(
        os.path.join(root_dir, "checkpoints_is/the_best_is.ckpt"),
        mode='max',
        monitor='inception_score',
        verbose=1,
        save_best_only=True)
    checkpoints_fid = ModelCheckpoint(
        os.path.join(root_dir, "checkpoints_fid/the_best_fid.ckpt"),
        mode='min',
        monitor='FID',
        verbose=1,
        save_best_only=True)
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
    inception_score_display = Display(layout=(1, 1),
                                      dpi='auto',
                                      show=show_figure,
                                      block_on_end=block_figure_on_end,
                                      filepath=[os.path.join(root_dir,
                                                             "inception_score/"
                                                             "inception_score_{epoch:04d}.png"),
                                                os.path.join(root_dir,
                                                             "inception_score/"
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
    fid_display = Display(layout=(1, 1),
                          dpi='auto',
                          show=show_figure,
                          block_on_end=block_figure_on_end,
                          filepath=[os.path.join(root_dir, "FID/"
                                                           "FID_{epoch:04d}.png"),
                                    os.path.join(root_dir, "FID/"
                                                           "FID_{epoch:04d}.pdf")],
                          monitor=[{'metrics': ['FID'],
                                    'type': 'line',
                                    'labels': ["FID"],
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
                             filepath=os.path.join(root_dir, "samples/samples_{epoch:04d}.png"),
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
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss',
                           'inception_score', 'inception_score_std', 'FID'],
                  callbacks=[loss_display, inception_score_display, fid_display,
                             sample_display, checkpoints_is, checkpoints_fid],
                  num_epochs=4,  # set to 100 for a full run
                  inception_metrics=[InceptionScore(), FID(data="cifar10")],
                  inception_metrics_freq=1,
                  # summary_freq=1,  # uncomment this for a full run
                  log_path=os.path.join(root_dir, "logs"),
                  random_state=random_seed(),
                  verbose=1)

    model.fit(x_train)
    filepath = os.path.join(root_dir, "checkpoints_fid/the_best_fid.ckpt")
    print("Reloading the latest model at: {}".format(filepath))
    model1 = TensorFlowModel.load_model(filepath)
    model1.inception_metrics = InceptionMetricList([InceptionScore(), FID(data="cifar10")])
    model1.num_epochs = 6
    model1.fit(x_train)
    print("Done!")


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_dcgan_image_saver():
    print("========== Test DCGAN with Image Saver ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_mnist()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 28, 28, 1]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/imagesaver/mnist")
    imgsaver = ImageSaver(freq=1,
                          filepath=os.path.join(root_dir, "mnist_{epoch:04d}.png"),
                          monitor={'metrics': 'x_samples',
                                   'num_samples': 100,
                                   'tile_shape': (10, 10),
                                   })

    model = DCGAN(model_name="DCGAN_MNIST",
                  num_z=10,  # set to 100 for a full run
                  img_size=(28, 28, 1),
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[imgsaver],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  log_path=os.path.join(root_dir, "logs"),
                  verbose=1)

    model.fit(x_train)

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/DCGAN/imagesaver/cifar10")
    imgsaver = ImageSaver(freq=1,
                          filepath=os.path.join(root_dir, "cifar10_{epoch:04d}.png"),
                          monitor={'metrics': 'x_samples',
                                   'num_samples': 100,
                                   'tile_shape': (10, 10),
                                   })

    model = DCGAN(model_name="DCGAN_CIFAR10",
                  num_z=10,  # set to 100 for a full run
                  img_size=(32, 32, 3),
                  batch_size=16,  # set to 64 for a full run
                  num_conv_layers=3,  # set to 3 for a full run
                  num_gen_feature_maps=4,  # set to 32 for a full run
                  num_dis_feature_maps=4,  # set to 32 for a full run
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[imgsaver],
                  num_epochs=4,  # set to 100 for a full run
                  random_state=random_seed(),
                  log_path=os.path.join(root_dir, "logs"),
                  verbose=1)

    model.fit(x_train)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_dcgan_mnist(show_figure=True, block_figure_on_end=True)
    # test_dcgan_fashion_mnist(show_figure=True, block_figure_on_end=True)
    # test_dcgan_save_and_load(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10_inception_score(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10_fid(show_figure=True, block_figure_on_end=True)
    # test_dcgan_cifar10_inception_metric(show_figure=True, block_figure_on_end=True)
    # test_dcgan_image_saver()
