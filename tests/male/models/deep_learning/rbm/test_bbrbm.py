from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male.configs import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.rbm import BernoulliBernoulliRBM


def test_bbrbm_cd(show_figure=False, block_figure_on_end=False):
    print("========== Test BernoulliBernoulliRBM using Contrastive Divergence ==========")

    np.random.seed(random_seed())

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(2, 2),
                               freq=1,
                               show=show_figure,
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
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 15,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (3, 5),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:1000],
                                       },
                                      ])

    model = BernoulliBernoulliRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, filter_display, hidden_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x)

    train_free_energy = model.get_free_energy(x_train).mean()
    test_free_energy = model.get_free_energy(x_test).mean()
    print("Train free energy = %.4f" % train_free_energy)
    print("Test free energy = %.4f" % test_free_energy)

    train_recon_loglik = model.get_reconstruction_loglik(x_train).mean()
    test_recon_loglik = model.get_reconstruction_loglik(x_test).mean()
    print("Train reconstruction likelihood = %.4f" % train_recon_loglik)
    print("Test reconstruction likelihood = %.4f" % test_recon_loglik)

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    test_pred_err = 1.0 - accuracy_score(y_test, clf.predict(x_test1))
    print("BBRBM->kNN: test error = %.4f" % test_pred_err)


def test_bbrbm_pcd(show_figure=False, block_figure_on_end=False):
    print("========== Test BernoulliBernoulliRBM using "
          "Persistent Contrastive Divergence ==========")

    np.random.seed(random_seed())

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(2, 2),
                               freq=1,
                               show=show_figure,
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
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 15,
                                       'disp_dim': (28, 28),
                                       'tile_shape': (3, 5),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             show=show_figure,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:1000],
                                       },
                                      ])

    model = BernoulliBernoulliRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        learning_method='pcd',
        num_pcd=5,
        num_chains=10,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err', 'free_energy', 'recon_loglik'],
        callbacks=[learning_display, filter_display, hidden_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x)

    train_free_energy = model.get_free_energy(x_train).mean()
    test_free_energy = model.get_free_energy(x_test).mean()
    print("Train free energy = %.4f" % train_free_energy)
    print("Test free energy = %.4f" % test_free_energy)

    train_recon_loglik = model.get_reconstruction_loglik(x_train).mean()
    test_recon_loglik = model.get_reconstruction_loglik(x_test).mean()
    print("Train reconstruction likelihood = %.4f" % train_recon_loglik)
    print("Test reconstruction likelihood = %.4f" % test_recon_loglik)

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    test_pred_err = 1.0 - accuracy_score(y_test, clf.predict(x_test1))
    print("BBRBM->kNN: test error = %.4f" % test_pred_err)


def test_bbrbm_csl():
    print("========== Test Conservative Sampling-based Likelihood (CSL) "
          "of BernoulliBernoulliRBM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    model = BernoulliBernoulliRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err'],
        random_state=random_seed(),
        verbose=1)

    model.fit(x_train)

    train_csl = model.get_loglik(x_train, method='csl',
                                 num_hidden_samples=100,
                                 num_steps=10).mean()
    print("Training log-likelihood computed using CSL = %.4f" % train_csl)

    test_csl = model.get_loglik(x_test, method='csl',
                                num_hidden_samples=100,
                                num_steps=10).mean()
    print("Testing log-likelihood computed using CSL = %.4f" % test_csl)


def test_bbrbm_generate_data(show_figure=False, block_figure_on_end=False):
    print("========== Test Data Generation of BernoulliBernoulliRBM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    model = BernoulliBernoulliRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err'],
        random_state=random_seed(),
        verbose=1)

    model.fit(x_train)

    num_samples = 25
    x_gen = model.generate_data(num_samples=num_samples)

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from male.utils.disp_utils import tile_raster_images

    img = tile_raster_images(x_gen, img_shape=(28, 28), tile_shape=(5, 5), tile_spacing=(1, 1),
                             scale_rows_to_unit_interval=False, output_pixel_vals=False)
    plt.figure()
    _ = plt.imshow(img, aspect='auto', cmap='Greys_r', interpolation='none')
    plt.title("Generate {} samples".format(num_samples))
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    if show_figure:
        plt.show(block=block_figure_on_end)


def test_bbrbm_logpartition():
    print("========== Test Computing log-partition function of BernoulliBernoulliRBM ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    model = BernoulliBernoulliRBM(
        num_hidden=15,
        num_visible=784,
        batch_size=100,
        num_epochs=4,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err'],
        random_state=random_seed(),
        verbose=1)

    model.fit(x_train)

    exact_logpart = model.get_logpartition(method='exact')
    print("Exact log-partition function = %.4f" % exact_logpart)

    test_exact_loglik = model.get_loglik(x_test, method='exact').mean()
    print("Exact log-likelihood of testing data = %.4f" % test_exact_loglik)

    test_csl_loglik = model.get_loglik(x_test, method='csl',
                                       num_hidden_samples=100,
                                       num_steps=100).mean()
    print("CSL log-likelihood of testing data = %.4f" % test_csl_loglik)


def test_bbrbm_pipeline():
    print("========== Test the pipeline of "
          "BernoulliBernoulliRBM followed by k-nearest-neighbors (kNN) ==========")

    np.random.seed(random_seed())

    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    estimators = [('rbm', BernoulliBernoulliRBM(num_hidden=15,
                                                num_visible=784,
                                                batch_size=100,
                                                num_epochs=4,
                                                momentum_method='sudden',
                                                weight_cost=2e-4,
                                                random_state=random_seed(),
                                                verbose=0)),
                  ('knn', KNeighborsClassifier(n_neighbors=1))]

    clf = Pipeline(estimators)

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_bbrbm_gridsearch():
    print("========== Tuning parameters for the pipeline of "
          "BernoulliBernoulliRBM followed by k-nearest-neighbors (kNN) ==========")

    np.random.seed(random_seed())

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_mnist()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    estimators = [('rbm', BernoulliBernoulliRBM(num_hidden=500,
                                                num_visible=784,
                                                batch_size=100,
                                                num_epochs=4,
                                                momentum_method='sudden',
                                                weight_cost=2e-4,
                                                random_state=random_seed(),
                                                verbose=0)),
                  ('knn', KNeighborsClassifier(n_neighbors=4))]

    params = dict(rbm__num_hidden=[10, 15],
                  rbm__batch_size=[64, 100],
                  knn__n_neighbors=[1, 2])

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = Pipeline(estimators)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1.0 - gs.best_score_, gs.best_params_))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_bbrbm_cd(show_figure=True, block_figure_on_end=True)
    # test_bbrbm_pcd(show_figure=True, block_figure_on_end=True)
    # test_bbrbm_csl()
    # test_bbrbm_generate_data(show_figure=True, block_figure_on_end=True)
    # test_bbrbm_logpartition()
    # test_bbrbm_pipeline()
    # test_bbrbm_gridsearch()
