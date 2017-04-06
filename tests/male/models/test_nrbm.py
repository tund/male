from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.rbm import NonnegativeRBM


def test_nrbm_mnist():
    np.random.seed(random_seed())
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(2, 2),
                               freq=1,
                               monitor=[{'metrics': ['recon_err', 'val_recon_err'],
                                         'type': 'line',
                                         'labels': ["training recon error", "validation recon error"],
                                         'title': "Reconstruction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        {'metrics': ['free_energy', 'val_free_energy'],
                                         'type': 'line',
                                         'title': "Free Energies",
                                         'xlabel': "epoch",
                                         'ylabel': "energy",
                                         },
                                        {'metrics': ['recon_loglik', 'val_recon_loglik'],
                                         'type': 'line',
                                         'labels': ["training recon loglik", "validation recon loglik"],
                                         'title': "Reconstruction Loglikelihoods",
                                         'xlabel': "epoch",
                                         'ylabel': "loglik",
                                         },
                                        {'metrics': ['recon_loglik', 'val_recon_loglik'],
                                         'type': 'line',
                                         'labels': ["training recon loglik", "validation recon loglik"],
                                         'title': "Reconstruction Loglikelihoods",
                                         'xlabel': "epoch",
                                         'ylabel': "loglik",
                                         },
                                        # {'metrics': ['loglik_csl', 'val_loglik_csl'],
                                        #  'type': 'line',
                                        #  'labels': ["training loglik (CSL)", "validation loglik (CSL)"],
                                        #  'title': "Loglikelihoods using CSL",
                                        #  'xlabel': "epoch",
                                        #  'ylabel': "loglik",
                                        #  },
                                        ])

    filter_display = Display(title="Receptive Fields",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    model = NonnegativeRBM(
        num_hidden=500,
        num_visible=784,
        batch_size=100,
        num_epochs=100,
        momentum_method='sudden',
        weight_cost=2e-4,
        nonnegative_type='barrier',
        nonnegative_cost=0.001,
        random_state=6789,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, filter_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x)

    print("Train free energy = %.4f" % model.get_free_energy(x_train).mean())
    print("Test free energy = %.4f" % model.get_free_energy(x_test).mean())

    print("Train reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_train).mean())
    print("Test reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_test).mean())

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    print("Error = %.4f" % (1 - accuracy_score(y_test, clf.predict(x_test1))))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_nrbm_mnist()
