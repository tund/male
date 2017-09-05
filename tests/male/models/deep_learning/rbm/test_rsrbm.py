from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.callbacks import Display
from male.models.deep_learning.rbm import ReplicatedSoftmaxRBM


def test_rsrbm_cd(block_figure_on_end=False):
    print("========== Test ReplicatedSoftmaxRBM using Contrastive Divergence ==========")

    np.random.seed(random_seed())

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_20newsgroups()
    # import os
    # from sklearn.datasets import load_svmlight_file
    # from male import data_dir
    # train_path = os.path.join(data_dir(), "20newsgroups/20news_bydate/libsvm/"
    #                                       "20news_bydate_5Kwordcount_in_entire_data_countfeat_train.libsvm")
    # test_path = os.path.join(data_dir(), "20newsgroups/20news_bydate/libsvm/"
    #                                      "20news_bydate_5Kwordcount_in_entire_data_countfeat_test.libsvm")
    # x_train, y_train = load_svmlight_file(train_path, n_features=5000)
    # x_test, y_test = load_svmlight_file(test_path, n_features=5000)
    # x_train = x_train.toarray()
    # x_test = x_test.toarray()
    #
    # x_train, y_train = demo.shuffle(x_train, y_train, randseed=random_seed())

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(2, 1),
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
                                         },
                                        ])

    filter_display = Display(title="Receptive Fields",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 10,
                                       'disp_dim': (25, 20),
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:200],
                                       },
                                      ])

    model = ReplicatedSoftmaxRBM(
        num_hidden=15,
        num_visible=5000,
        batch_size=32,
        num_epochs=4,
        # sparse_weight=0.3,
        # sparse_level=0.1,
        learning_rate=0.001,
        learning_rate_hidden=0.00001,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err', 'free_energy'],
        callbacks=[learning_display, filter_display, hidden_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x)

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    test_pred_err = 1.0 - accuracy_score(y_test, clf.predict(x_test1))
    print("RSRBM->kNN: test error = %.4f" % test_pred_err)


def test_rsrbm_pipeline():
    print("========== Test the pipeline of "
          "ReplicatedSoftmaxRBM followed by k-nearest-neighbors (kNN) ==========")

    np.random.seed(random_seed())

    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_20newsgroups()

    estimators = [('rbm', ReplicatedSoftmaxRBM(num_hidden=15,
                                               num_visible=5000,
                                               batch_size=32,
                                               num_epochs=4,
                                               learning_rate=0.001,
                                               learning_rate_hidden=0.00001,
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


def test_rsrbm_gridsearch():
    print("========== Tuning parameters for the pipeline of "
          "ReplicatedSoftmaxRBM followed by k-nearest-neighbors (kNN) ==========")

    np.random.seed(random_seed())

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    from sklearn.neighbors import KNeighborsClassifier

    (x_train, y_train), (x_test, y_test) = demo.load_20newsgroups()

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    estimators = [('rbm', ReplicatedSoftmaxRBM(num_hidden=15,
                                               num_visible=5000,
                                               batch_size=32,
                                               num_epochs=2,
                                               learning_rate=0.001,
                                               learning_rate_hidden=0.00001,
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


@pytest.mark.skip(reason="Very long running time.")
def test_rsrbm_cd_on_full_20newsgroups_dataset(block_figure_on_end=False):
    print("========== Test ReplicatedSoftmaxRBM using Contrastive Divergence ==========")

    np.random.seed(random_seed())

    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    import os
    from sklearn.datasets import load_svmlight_file
    from male import data_dir
    train_path = os.path.join(data_dir(), "20newsgroups/20news_bydate/libsvm/"
                                          "20news_bydate_5Kwordcount_in_entire_data_countfeat_train.libsvm")
    test_path = os.path.join(data_dir(), "20newsgroups/20news_bydate/libsvm/"
                                         "20news_bydate_5Kwordcount_in_entire_data_countfeat_test.libsvm")
    x_train, y_train = load_svmlight_file(train_path, n_features=5000)
    x_test, y_test = load_svmlight_file(test_path, n_features=5000)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    x_train, y_train = demo.shuffle(x_train, y_train, randseed=random_seed())

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    learning_display = Display(title="Learning curves",
                               dpi='auto',
                               layout=(2, 1),
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
                                         },
                                        ])

    filter_display = Display(title="Receptive Fields",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['filters'],
                                       'title': "Receptive Fields",
                                       'type': 'img',
                                       'num_filters': 10,
                                       'disp_dim': (25, 20),
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    hidden_display = Display(title="Hidden Activations",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(8, 8),
                             freq=1,
                             block_on_end=block_figure_on_end,
                             monitor=[{'metrics': ['hidden_activations'],
                                       'title': "Hidden Activations",
                                       'type': 'img',
                                       'data': x_train[:200],
                                       },
                                      ])

    model = ReplicatedSoftmaxRBM(
        num_hidden=100,
        num_visible=5000,
        batch_size=128,
        num_epochs=100,
        # sparse_weight=0.3,
        # sparse_level=0.1,
        learning_rate=0.01,
        learning_rate_hidden=0.0001,
        momentum_method='sudden',
        weight_cost=2e-4,
        metrics=['recon_err', 'free_energy'],
        callbacks=[learning_display, filter_display, hidden_display],
        cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
        random_state=random_seed(),
        verbose=1)

    model.fit(x)

    x_train1 = model.transform(x_train)
    x_test1 = model.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(x_train1, y_train)

    test_pred_err = 1.0 - accuracy_score(y_test, clf.predict(x_test1))
    print("RSRBM->kNN: test error = %.4f" % test_pred_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_rsrbm_cd(block_figure_on_end=True)
    # test_rsrbm_pipeline()
    # test_rsrbm_gridsearch()
    # test_rsrbm_cd_on_full_20newsgroups_dataset(block_figure_on_end=True)
