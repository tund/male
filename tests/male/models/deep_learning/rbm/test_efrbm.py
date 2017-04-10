from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from male.models.deep_learning.rbm import BernoulliBernoulliRBM

from male.callbacks import Display
from male.models.deep_learning.rbm import EFRBM
from male.models.deep_learning.rbm import BernoulliBernoulliRBM
from data_config import data_config

SUFFICIENT_STATISTICS_DIM = {'binary':1, 'categorical': 10, 'continuous':2}
# from sytem_config import system_config
# SYSINFO = system_config()
#
# if SYSINFO['display']==False:
#     import matplotlib
#     # matplotlib.use('Agg')
#     matplotlib.use('Qt4Agg')

# from data_config import data_config
data_info = data_config('MNIST')
data_folder = data_info['data_folder']
temp_folder = data_info['temp_folder']

x_train, y_train = load_svmlight_file('%s/mnist' % data_folder, n_features=784)
x_test, y_test = load_svmlight_file('%s/mnist.t' % data_folder, n_features=784)

num_train = 1000
num_test = 500


x_train = x_train[:num_train]
y_train = y_train[:num_train]
x_test = x_test[:num_test]
y_test = y_test[:num_test]


x_train = x_train.toarray() / 255.0
idx_train = np.random.permutation(x_train.shape[0])
x_train = x_train[idx_train]
y_train = y_train[idx_train]

x_test = x_test.toarray() / 255.0
idx_test = np.random.permutation(x_test.shape[0])
x_test = x_test[idx_test]
y_test = y_test[idx_test]

def test_efrbm_mnist(visible_layer_type = 'binary', hidden_layer_type = 'continuous'):
    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(1, 2),
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
                                         }
                                        ])

    # filter_display = Display(title="Receptive Fields",
    #                          # dpi='auto',
    #                          dpi=None,
    #                          layout=(1, 1),
    #                          figsize=(8, 8),
    #                          freq=1,
    #                          monitor=[{'metrics': ['filters'],
    #                                    'title': "Receptive Fields",
    #                                    'type': 'img',
    #                                    'num_filters': 100,
    #                                    'disp_dim': (28, 28),
    #                                    'tile_shape': (10, 10),
    #                                    },
    #                                   ])

    gen_display = Display(title="Generated data",
                             # dpi='auto',
                             dpi=None,
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             monitor=[{'metrics': ['generated_data'],
                                       'title': "Generated data",
                                       'type': 'img',
                                       'num_filters': 100,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (10, 10),
                                       },
                                      ])

    recon_display = Display(title="Reconstructed data",
                          # dpi='auto',
                          dpi=None,
                          layout=(1, 1),
                          figsize=(8, 8),
                          freq=1,
                          monitor=[{'metrics': ['reconstruction'],
                                    'title': "Reconstructed data",
                                    'type': 'img',
                                    'data': x_train,
                                    'num_filters': 100,
                                    'disp_dim': (28, 28),
                                    'tile_shape': (10, 10),
                                    },
                                   ])

    # model_path = "%s/EFDBM/numtrain%d" % (temp_folder, x_train.shape[0])
    # if os.path.isdir(model_path) == False:
    #     os.makedirs(model_path)

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
        Gaussian_layer_trainable_sigmal2 = Gaussian_layer_trainable_sigmal2,
        num_hidden=500,
        num_visible=784,
        batch_size=100,
        num_epochs=50,
        momentum_method='sudden',
        weight_cost=2e-4,
        # sparse_weight = 0.1,
        random_state=6789,
        w_init=w_init,
        learning_rate=learning_rate,
        metrics=['recon_err', 'free_energy'],
        callbacks=[learning_display, recon_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
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
    # pytest.main([__file__])
    test_efrbm_mnist(visible_layer_type='binary', hidden_layer_type='binary')


