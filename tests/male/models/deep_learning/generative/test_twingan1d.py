from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest

from male import random_seed
from male.callbacks import Display
from male.models.distribution import GMM1D
from male.models.distribution import Uniform1D
from male.models.distribution import Gaussian1D
from male.models.deep_learning.generative import TwinGAN1D


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_twingan1d_gaussian1d(block_figure_on_end=False):
    loss_display = Display(layout=(1, 3),
                           dpi='auto',
                           title='Loss',
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
                                    {'metrics': ['d1x', 'd2x'],
                                     'type': 'line',
                                     'labels': ["d1x_mean", "d2x_mean"],
                                     'title': "Discriminative scores",
                                     'xlabel': "epoch",
                                     'ylabel': "score",
                                     },
                                    ])
    distribution_display = Display(layout=(1, 1),
                                   dpi='auto',
                                   freq=10,
                                   title='Histogram',
                                   block_on_end=block_figure_on_end,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "Histogram of TwinGAN1D",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])
    avg_distribution_display = Display(layout=(1, 1),
                                       dpi='auto',
                                       freq=10,
                                       title='Average Histogram',
                                       block_on_end=block_figure_on_end,
                                       monitor=[{'metrics': ['avg_distribution'],
                                                 'type': 'hist',
                                                 'title': "Averaged Histogram of TwinGAN1D",
                                                 'xlabel': "Data values",
                                                 'ylabel': "Probability density",
                                                 },
                                                ])

    model = TwinGAN1D(data=Gaussian1D(mu=4.0, sigma=0.5),
                      generator=Uniform1D(low=-8.0, high=8.0),
                      alpha=1.0,
                      beta=1.0,
                      num_z=10,  # increase this to 100 for a full run
                      num_epochs=4,  # increase this to 1000 for a full run
                      hidden_size=16,  # increase this to 128 for a full run
                      batch_size=16,
                      minibatch_discriminator=False,
                      loglik_freq=1,
                      generator_learning_rate=0.0001,
                      discriminator_learning_rate=0.0001,
                      metrics=['d1x', 'd2x', 'd1_loss', 'd2_loss', 'd_loss', 'g_loss', 'loglik'],
                      callbacks=[loss_display, distribution_display, avg_distribution_display],
                      random_state=random_seed(),
                      verbose=1)
    model.fit()


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_twingan1d_gmm1d(block_figure_on_end=False):
    loss_display = Display(layout=(1, 3),
                           dpi='auto',
                           title='Loss',
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
                                    {'metrics': ['d1x', 'd2x'],
                                     'type': 'line',
                                     'labels': ["d1x_mean", "d2x_mean"],
                                     'title': "Discriminative scores",
                                     'xlabel': "epoch",
                                     'ylabel': "score",
                                     },
                                    ])
    distribution_display = Display(layout=(1, 1),
                                   dpi='auto',
                                   freq=20,
                                   title='Histogram',
                                   block_on_end=block_figure_on_end,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "Histogram of TwinGAN1D",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])
    avg_distribution_display = Display(layout=(1, 1),
                                       dpi='auto',
                                       freq=10,
                                       title='Average Histogram',
                                       block_on_end=block_figure_on_end,
                                       monitor=[{'metrics': ['avg_distribution'],
                                                 'type': 'hist',
                                                 'title': "Averaged Histogram of TwinGAN1D",
                                                 'xlabel': "Data values",
                                                 'ylabel': "Probability density",
                                                 },
                                                ])

    model = TwinGAN1D(data=GMM1D(pi=[0.55, 0.45], mu=[1.0, 4.0], sigma=[0.2, 0.5]),
                      generator=Uniform1D(low=-8.0, high=8.0),
                      alpha=1.0,
                      beta=1.0,
                      num_z=10,  # increase this to 100 for a full run
                      num_epochs=4,  # increase this to 1000 for a full run
                      hidden_size=16,  # increase this to 128 for a full run
                      batch_size=16,
                      loglik_freq=1,
                      generator_learning_rate=0.001,
                      discriminator_learning_rate=0.001,
                      metrics=['d1x', 'd2x', 'd1_loss', 'd2_loss', 'd_loss', 'g_loss', 'loglik'],
                      callbacks=[loss_display, distribution_display, avg_distribution_display],
                      random_state=random_seed(),
                      verbose=1)
    model.fit()


if __name__ == '__main__':
    pytest.main([__file__])
    # test_twingan1d_gaussian1d(block_figure_on_end=True)
    # test_twingan1d_gmm1d(block_figure_on_end=True)
