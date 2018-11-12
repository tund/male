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
from male.models.deep_learning.generative import WGAN_GP_ResNet


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_wgan_gp_resnet_cifar10(show_figure=False, block_figure_on_end=False):
    print("========== Test WGAN-GP-ResNet on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    root_dir = os.path.join(model_dir(), "male/WGAN-GP-ResNet/CIFAR10")
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
                             monitor=[{'metrics': ['x_samples'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_samples': 100,
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = WGAN_GP_ResNet(model_name="WGAN_GP_ResNet_CIFAR10",
                           num_z=10,  # set to 100 for a full run
                           img_size=(32, 32, 3),
                           batch_size=16,  # set to 64 for a full run
                           g_blocks=('up', 'up', 'up'),
                           d_blocks=('down', 'down', None, None),
                           num_gen_feature_maps=4,  # set to 32 for a full run
                           num_dis_feature_maps=4,  # set to 32 for a full run
                           metrics=['d_loss', 'g_loss'],
                           callbacks=[loss_display, sample_display],
                           num_epochs=2,  # set to 100 for a full run
                           random_state=random_seed(),
                           log_path=os.path.join(root_dir, "logs"),
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
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_wgan_gp_resnet_cifar10_inception_metric(show_figure=False, block_figure_on_end=False):
    print("========== Test WGAN-GP-ResNet with Inception Score and FID on CIFAR10 data ==========")

    np.random.seed(random_seed())

    num_data = 128
    (x_train, y_train), (x_test, y_test) = demo.load_cifar10()
    x_train = x_train[:num_data].astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.
    x_test = x_test.astype(np.float32).reshape([-1, 32, 32, 3]) / 0.5 - 1.

    # uncomment for full run
    '''
    import pickle
    from male.configs import data_dir
    tmp = pickle.load(open(os.path.join(data_dir(), "cifar10/cifar10_train.pkl"), "rb"))
    x_train = tmp['data'].astype(np.float32).reshape(
        [-1, 32, 32, 3]) / 127.5 - 1.
    '''

    root_dir = os.path.join(model_dir(), "male/WGAN-GP-ResNet/CIFAR10")
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

    model = WGAN_GP_ResNet(model_name="WGAN_GP_ResNet_CIFAR10",
                           num_z=8,  # set to 128 for a full run
                           img_size=(32, 32, 3),
                           batch_size=64,  # set to 64 for a full run
                           g_blocks=('up', 'up', 'up'),
                           d_blocks=('down', 'down', None, None),
                           num_gen_feature_maps=8,  # set to 128 for a full run
                           num_dis_feature_maps=8,  # set to 128 for a full run
                           metrics=['d_loss', 'g_loss',
                                    'inception_score', 'inception_score_std', 'FID'],
                           callbacks=[loss_display, inception_score_display, fid_display,
                                      sample_display, checkpoints_is, checkpoints_fid],
                           num_epochs=2,  # set to 500 for a full run
                           inception_metrics=[InceptionScore(), FID(data='cifar10')],
                           inception_metrics_freq=1,
                           num_inception_samples=100,  # set to 50000 for a full run
                           # summary_freq=1,  # uncomment this for a full run
                           log_path=os.path.join(root_dir, 'logs'),
                           random_state=random_seed(),
                           verbose=1)

    model.fit(x_train)
    filepath = os.path.join(root_dir, 'checkpoints_fid/the_best_fid.ckpt')
    print('Reloading the latest model at: {}'.format(filepath))
    model1 = TensorFlowModel.load_model(filepath)
    model1.inception_metrics = InceptionMetricList([InceptionScore(), FID(data='cifar10')])
    model1.num_epochs = 4
    model1.fit(x_train)
    print('Done!')


if __name__ == '__main__':
    pytest.main([__file__])
    # test_wgan_gp_resnet_cifar10(show_figure=True, block_figure_on_end=True)
    # test_wgan_gp_resnet_cifar10_inception_metric(show_figure=True, block_figure_on_end=True)
