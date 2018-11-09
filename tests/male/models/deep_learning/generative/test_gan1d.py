from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import pytest

from male.configs import model_dir
from male.configs import random_seed
from male.callbacks import Display
from male.models.distributions import GMM1D
from male.models.distributions import Uniform1D
from male.models.distributions import Gaussian1D
from male.models.deep_learning.generative import GAN1D


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_gan1d_gaussian1d(show_figure=False, block_figure_on_end=False):
    print("========== Test GAN on 1D data generated from a Gaussian distribution ==========")

    loss_display = Display(layout=(2, 1),
                           dpi='auto',
                           title='Loss',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['loglik'],
                                     'type': 'line',
                                     'labels': ["Log-likelihood"],
                                     'title': "Evaluation",
                                     'xlabel': "epoch",
                                     'ylabel': "loglik",
                                     },
                                    ])
    distribution_display = Display(layout=(1, 1),
                                   dpi='auto',
                                   freq=1,
                                   title='Histogram',
                                   show=show_figure,
                                   block_on_end=block_figure_on_end,
                                   # filepath=[os.path.join(model_dir(),
                                   #                        "GAN1D/samples/hist_{epoch:04d}.png")],
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "Histogram of GAN1D",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])
    avg_distribution_display = Display(layout=(1, 1),
                                       dpi='auto',
                                       freq=1,
                                       title='Average Histogram',
                                       show=show_figure,
                                       block_on_end=block_figure_on_end,
                                       monitor=[{'metrics': ['avg_distribution'],
                                                 'type': 'hist',
                                                 'title': "Averaged Histogram of GAN1D",
                                                 'xlabel': "Data values",
                                                 'ylabel': "Probability density",
                                                 },
                                                ])

    model = GAN1D(data=Gaussian1D(mu=4.0, sigma=0.5),
                  generator=Uniform1D(low=-8.0, high=8.0),
                  num_z=8,  # set to 100 for a full run
                  num_epochs=4,  # set to 355 for a full run
                  hidden_size=4,  # set to 128 for a full run
                  batch_size=8,  # set to 128 for a full run
                  minibatch_discriminator=False,
                  loglik_freq=1,
                  generator_learning_rate=0.0001,
                  discriminator_learning_rate=0.0001,
                  metrics=['d_loss', 'g_loss', 'loglik'],
                  callbacks=[loss_display, distribution_display, avg_distribution_display],
                  random_state=random_seed(),
                  verbose=1)
    model.fit()


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_gan1d_gmm1d(show_figure=False, block_figure_on_end=False):
    print("========== Test GAN on 1D data generated from a Gaussian Mixture Model ==========")

    loss_display = Display(layout=(2, 1),
                           dpi='auto',
                           title='Loss',
                           show=show_figure,
                           block_on_end=block_figure_on_end,
                           monitor=[{'metrics': ['d_loss', 'g_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss", "generator loss"],
                                     'title': "Losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['loglik'],
                                     'type': 'line',
                                     'labels': ["Log-likelihood"],
                                     'title': "Evaluation",
                                     'xlabel': "epoch",
                                     'ylabel': "loglik",
                                     },
                                    ])
    distribution_display = Display(layout=(1, 1),
                                   dpi='auto',
                                   freq=1,
                                   title='Histogram',
                                   show=show_figure,
                                   block_on_end=block_figure_on_end,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "Histogram of GAN1D",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])
    avg_distribution_display = Display(layout=(1, 1),
                                       dpi='auto',
                                       freq=1,
                                       title='Average Histogram',
                                       show=show_figure,
                                       block_on_end=block_figure_on_end,
                                       monitor=[{'metrics': ['avg_distribution'],
                                                 'type': 'hist',
                                                 'title': "Averaged Histogram of GAN1D",
                                                 'xlabel': "Data values",
                                                 'ylabel': "Probability density",
                                                 },
                                                ])

    model = GAN1D(data=GMM1D(pi=[0.55, 0.45], mu=[1.0, 4.0], sigma=[0.2, 0.5]),
                  generator=Uniform1D(low=-8.0, high=8.0),
                  num_z=8,  # set to 100 for a full run
                  num_epochs=4,  # set to 1000 for a full run
                  hidden_size=4,  # set to 128 for a full run
                  batch_size=8,  # set to 128 for a full run
                  loglik_freq=1,
                  generator_learning_rate=0.0001,
                  discriminator_learning_rate=0.0001,
                  metrics=['d_loss', 'g_loss', 'loglik'],
                  callbacks=[loss_display, distribution_display, avg_distribution_display],
                  random_state=random_seed(),
                  verbose=1)
    model.fit()


if __name__ == '__main__':
    pytest.main([__file__])
    # test_gan1d_gaussian1d(show_figure=True, block_figure_on_end=True)
    # test_gan1d_gmm1d(show_figure=True, block_figure_on_end=True)
