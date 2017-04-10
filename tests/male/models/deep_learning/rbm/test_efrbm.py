from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.rbm import EFRBM

SUFFICIENT_STATISTICS_DIM = {'binary': 1, 'categorical': 10, 'continuous': 2}


def test_efrbm_mnist(visible_layer_type='binary', hidden_layer_type='continuous',
                     block_figure_on_end=False):
    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    num_train = 1000
    num_test = 500

    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(1, 2),
                               freq=1,
                               block_on_end=block_figure_on_end,
                               monitor=[{'metrics': ['recon_err', 'val_recon_err'],
                                         'type': 'line',
                                         'labels': ["training recon error",
                                                    "validation recon error"],
                                         'title': "Reconstruction Errors",
                                         'xlabel': "epoch",
                                         'ylabel': "error",
                                         },
                                        {'metrics': ['free_energy', 'val_free_energy'],
                                         'type': 'line',
                                         'title': "Free Energies",
                                         'xlabel': "epoch",
                                         'ylabel': "energy",
                                         }
                                        ])

    gen_display = Display(title="Generated data",
                          dpi='auto',
                          layout=(1, 1),
                          figsize=(8, 8),
                          freq=1,
                          block_on_end=block_figure_on_end,
                          monitor=[{'metrics': ['generated_data'],
                                    'title': "Generated data",
                                    'type': 'img',
                                    'num_filters': 100,
                                    'disp_dim': (28, 28),
                                    'tile_shape': (10, 10),
                                    },
                                   ])

    recon_display = Display(title="Reconstructed data",
                            dpi='auto',
                            layout=(1, 1),
                            figsize=(8, 8),
                            freq=1,
                            block_on_end=block_figure_on_end,
                            monitor=[{'metrics': ['reconstruction'],
                                      'title': "Reconstructed data",
                                      'type': 'img',
                                      'data': x_train,
                                      'num_filters': 100,
                                      'disp_dim': (28, 28),
                                      'tile_shape': (10, 10),
                                      },
                                     ])

    suf_stat_dim_vis = SUFFICIENT_STATISTICS_DIM[visible_layer_type]
    suf_stat_dim_hid = SUFFICIENT_STATISTICS_DIM[hidden_layer_type]
    w_init = 0.1
    learning_rate = 0.1,
    Gaussian_layer_trainable_sigmal2 = True
    if visible_layer_type == 'continous' or hidden_layer_type == 'continuous':
        Gaussian_layer_trainable_sigmal2 = False
        w_init = 0.001
        learning_rate = 0.01

    model = EFRBM(
        suf_stat_dim_vis=suf_stat_dim_vis,
        visible_layer_type=visible_layer_type,
        suf_stat_dim_hid=suf_stat_dim_hid,
        hidden_layer_type=hidden_layer_type,
        Gaussian_layer_trainable_sigmal2=Gaussian_layer_trainable_sigmal2,
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=5,
        momentum_method='sudden',
        weight_cost=2e-4,
        w_init=w_init,
        learning_rate=learning_rate,
        metrics=['recon_err', 'free_energy'],
        callbacks=[learning_display, recon_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x)

    print("Running Logistic Regression...")

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = LogisticRegression()
    clf.fit(x_train1, y_train)

    y_test_pred = clf.predict(x_test1)

    print("Error = %.4f" % (1 - accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_efrbm_mnist(visible_layer_type='binary', hidden_layer_type='binary',
    #                  block_figure_on_end=True)
