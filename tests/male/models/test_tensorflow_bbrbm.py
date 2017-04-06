from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.rbm import BernoulliBernoulliTensorFlowRBM


def test_bbtfrbm_mnist():
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train = x_train.astype(np.float32) / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test.astype(np.float32) / 255.0
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
                                         },
                                        {'metrics': ['recon_loglik', 'val_recon_loglik'],
                                         'type': 'line',
                                         'labels': ["training recon loglik",
                                                    "validation recon loglik"],
                                         'title': "Reconstruction Loglikelihoods",
                                         'xlabel': "epoch",
                                         'ylabel': "loglik",
                                         },
                                        {'metrics': ['recon_loglik', 'val_recon_loglik'],
                                         'type': 'line',
                                         'labels': ["training recon loglik",
                                                    "validation recon loglik"],
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

    model = BernoulliBernoulliTensorFlowRBM(
        num_hidden=500,
        num_visible=784,
        batch_size=100,
        num_epochs=2,
        momentum_method='sudden',
        # sparse_weight=10.0,
        weight_cost=2e-4,
        random_state=6789,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, filter_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        verbose=1)

    model.fit(x)

    print("Train free energy = %.4f" % model.get_free_energy(x_train).mean())
    print("Test free energy = %.4f" % model.get_free_energy(x_test).mean())

    print(
        "Train reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_train).mean())
    print("Test reconstruction likelihood = %.4f" % model.get_reconstruction_loglik(x_test).mean())

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    print("Error = %.4f" % (1 - accuracy_score(y_test, clf.predict(x_test1))))


def test_bbtfrbm_mnist_csl():
    '''TODO: Fix the code to test TensorFlow Bernoulli-Bernoulli RBM
    '''
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    model = BernoulliBernoulliTensorFlowRBM(
        num_hidden=500,
        num_visible=784,
        batch_size=100,
        num_epochs=10,
        momentum_method='sudden',
        weight_cost=2e-4,
        random_state=6789,
        metrics=['recon_err'],
        verbose=1)

    model.fit(x_train)

    print("Training log-likelihood computed using CSL = %.4f" %
          model.get_loglik(x_train, method='csl',
                           num_hidden_samples=10000,
                           num_steps=100).mean())
    print("Testing log-likelihood computed using CSL = %.4f" %
          model.get_loglik(x_test, method='csl',
                           num_hidden_samples=10000,
                           num_steps=100).mean())


def test_bbtfrbm_mnist_generate_data():
    '''TODO: Fix the code to test TensorFlow Bernoulli-Bernoulli RBM
    '''
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    model = BernoulliBernoulliTensorFlowRBM(
        num_hidden=500,
        num_visible=784,
        batch_size=100,
        num_epochs=10,
        momentum_method='sudden',
        weight_cost=2e-4,
        random_state=6789,
        metrics=['recon_err'],
        verbose=1)

    model.fit(x_train)

    num_samples = 100
    x_gen = model.generate_data(num_samples=num_samples)

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from male.utils.disp_utils import tile_raster_images

    img = tile_raster_images(x_gen, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1),
                             scale_rows_to_unit_interval=False, output_pixel_vals=False)
    plt.figure()
    _ = plt.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
    plt.title("Generate {} samples".format(num_samples))
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_bbtfrbm_mnist_logpartition():
    '''TODO: Fix the code to test TensorFlow Bernoulli-Bernoulli RBM
    '''
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x_train /= 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test /= 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    model = BernoulliBernoulliTensorFlowRBM(
        num_hidden=5,
        num_visible=784,
        batch_size=100,
        num_epochs=10,
        momentum_method='sudden',
        weight_cost=2e-4,
        random_state=6789,
        metrics=['recon_err'],
        verbose=1)

    model.fit(x_train)

    print("Log-partition function = %.4f"
          % model.get_logpartition(method='exact'))
    print("Exact log-likelihood of testing data = %.4f"
          % model.get_loglik(x_test, method='exact').mean())
    print("CSL log-likelihood of testing data = %.4f" %
          model.get_loglik(x_test, method='csl',
                           num_hidden_samples=1000,
                           num_steps=1000).mean())


def test_bbtfrbm_mnist_gridsearch():
    '''TODO: Fix the code to test TensorFlow Bernoulli-Bernoulli RBM
    '''
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    from sklearn.neighbors import KNeighborsClassifier

    np.random.seed(random_seed())

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

    estimators = [('rbm', BernoulliBernoulliTensorFlowRBM(num_hidden=500,
                                                          num_visible=784,
                                                          batch_size=100,
                                                          num_epochs=10,
                                                          momentum_method='sudden',
                                                          weight_cost=2e-4,
                                                          random_state=6789,
                                                          verbose=1)),
                  ('knn', KNeighborsClassifier(n_neighbors=4))]

    params = dict(rbm__num_hidden=[10, 20, 50],
                  rbm__batch_size=[32, 64, 100],
                  knn__n_neighbors=[1, 4, 10])

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = Pipeline(estimators)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    # [Parallel(n_jobs=-1)]: Done  27 out of  27 | elapsed:  4.3min finished
    # Best score 0.916 @ params {'rbm__num_hidden': 50, 'rbm__batch_size': 64, 'knn__n_neighbors': 4}


if __name__ == '__main__':
    pytest.main([__file__])
    # test_bbtfrbm_mnist()
    # test_bbtfrbm_mnist_csl()
    # test_bbtfrbm_mnist_generate_data()
    # test_bbtfrbm_mnist_logpartition()
    # test_bbtfrbm_mnist_gridsearch()
