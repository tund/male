from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male.callbacks import Display
from male.models.distribution import GMM1D
from male.models.distribution import Uniform1D
from male.models.distribution import Gaussian1D
from male.models.deep_learning.generative import GRN1D


def test_grn1d_gaussian1d():
    loss_display = Display(layout=(2, 1),
                           monitor=[{'metrics': ['d_loss', 'g_loss', 'a_loss_1'],
                                     'type': 'line',
                                     'labels': ["discriminator loss",
                                                "generator loss",
                                                "auxiliary_1 loss"],
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
                                   freq=20,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "Histograms of GRN1D",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])

    avg_distribution_display = Display(layout=(1, 1),
                                       freq=20,
                                       monitor=[{'metrics': ['avg_distribution'],
                                                 'type': 'hist',
                                                 'title': "Average Histograms of GRN1D",
                                                 'xlabel': "Data values",
                                                 'ylabel': "Probability density",
                                                 },
                                                ])

    model = GRN1D(data=Gaussian1D(mu=4.0, sigma=0.5),
                  generator=Uniform1D(low=-8, high=8.0),
                  aux_discriminators=[Uniform1D(low=-6, high=14)],
                  aux_coeffs=[0.5],
                  num_epochs=5000,
                  hidden_size=20,
                  batch_size=12,
                  aux_batch_size=10,
                  discriminator_learning_rate=0.001,
                  generator_learning_rate=0.001,
                  aux_learning_rate=0.001,
                  loglik_freq=10,
                  metrics=['d_loss', 'g_loss', 'a_loss_1', 'loglik'],
                  callbacks=[loss_display, distribution_display, avg_distribution_display],
                  random_state=42,
                  verbose=1)
    model.fit()


def test_grn1d_gmm1d():
    loss_display = Display(layout=(2, 1),
                           monitor=[{'metrics': ['d_loss', 'g_loss', 'a_loss_1'],
                                     'type': 'line',
                                     'labels': ["discriminator loss",
                                                "generator loss",
                                                "auxiliary_1 loss"],
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
                                   freq=20,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "1D Generative Regularized Network",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])

    model = GRN1D(data=GMM1D(pi=[0.55, 0.45], mu=[1.0, 4.0], sigma=[0.2, 0.5]),
                  generator=Uniform1D(low=-8, high=8.0),
                  aux_discriminators=[Uniform1D(low=-6, high=14)],
                  aux_coeffs=[0.5],
                  num_epochs=5000,
                  hidden_size=20,
                  batch_size=12,
                  aux_batch_size=10,
                  discriminator_learning_rate=0.001,
                  generator_learning_rate=0.001,
                  aux_learning_rate=0.001,
                  loglik_freq=10,
                  metrics=['d_loss', 'g_loss', 'a_loss_1', 'loglik'],
                  callbacks=[loss_display, distribution_display],
                  random_state=42,
                  verbose=1)
    model.fit()


if __name__ == '__main__':
    # pytest.main([__file__])
    test_grn1d_gaussian1d()
    # test_grn1d_gmm1d()
