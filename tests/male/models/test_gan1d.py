from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male.callbacks import Display
from male.models.distribution import Uniform1D
from male.models.distribution import Gaussian1D
from male.models.deep_learning.generative import GAN1D


def test1():
    loss_display = Display(layout=(2, 1),
                           monitor=[{'metrics': ['d_loss'],
                                     'type': 'line',
                                     'labels': ["discriminator loss"],
                                     'title': "Discriminator loss",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['g_loss'],
                                     'type': 'line',
                                     'labels': ["generator loss"],
                                     'title': "Generator loss",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    ])
    distribution_display = Display(layout=(1, 1),
                                   freq=20,
                                   monitor=[{'metrics': ['distribution'],
                                             'type': 'hist',
                                             'title': "1D Generative Adversarial Network",
                                             'xlabel': "Data values",
                                             'ylabel': "Probability density",
                                             },
                                            ])

    model = GAN1D(data=Gaussian1D(mu=4.0, sigma=0.5),
                  generator=Uniform1D(low=-8.0, high=8.0),
                  num_epochs=620,
                  hidden_size=4,
                  batch_size=12,
                  metrics=['d_loss', 'g_loss'],
                  callbacks=[loss_display, distribution_display],
                  verbose=1)
    model.fit()


if __name__ == '__main__':
    # pytest.main([__file__])
    test1()
