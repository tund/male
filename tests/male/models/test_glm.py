from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn import metrics
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male.models.linear import GLM
from male.callbacks import Display
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_glm_check_grad():
    # <editor-fold desc="Binary classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = GLM(model_name="checkgrad_glm_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_logit",
                task='classification',
                link='logit',  # link function
                loss='logit',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = GLM(model_name="checkgrad_glm_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_softmax",
                task='classification',
                link='softmax',  # link function
                loss='softmax',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression">
    eps = 1e-6
    num_data = 10
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    model = GLM(model_name="checkgrad_glm_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.0,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.0,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps

    model = GLM(model_name="checkgrad_glm_quad",
                task='regression',
                link='linear',  # link function
                loss='quadratic',  # loss function
                l2_penalty=0.1,  # ridge regularization
                l1_penalty=0.01,  # Lasso regularization
                l1_smooth=1E-5,  # smoothing for Lasso regularization
                l1_method='pseudo_huber',  # approximation method for L1-norm
                random_state=6789)
    assert model.check_grad(x, y) < eps
    # </editor-fold>


def test_glm_mnist_logit():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    idx_train = np.where(np.uint8(y_train == 0) | np.uint8(y_train == 1))[0]
    print("# training samples = {}".format(len(idx_train)))
    x_train = x_train.toarray() / 255.0
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    idx_test = np.where(np.uint8(y_test == 0) | np.uint8(y_test == 1))[0]
    print("# testing samples = {}".format(len(idx_test)))
    x_test = x_test.toarray() / 255.0
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    clf = GLM(model_name="mnist_glm_logit",
              l1_penalty=0.0,
              l2_penalty=0.0,
              random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = GLM(model_name="mnist_glm_logit",
              optimizer='sgd',
              l1_penalty=0.0,
              l2_penalty=0.0,
              random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_glm_mnist_softmax():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    clf = GLM(model_name="mnist_glm_softmax",
              link='softmax',
              loss='softmax',
              random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = GLM(model_name="mnist_glm_softmax",
              optimizer='sgd',
              link='softmax',
              loss='softmax',
              random_state=6789)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_glm_mnist_logit_gridsearch():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    idx_train = np.where(np.uint8(y_train == 0) | np.uint8(y_train == 1))[0]
    print("# training samples = {}".format(len(idx_train)))
    x_train = x_train.toarray() / 255.0
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    idx_test = np.where(np.uint8(y_test == 0) | np.uint8(y_test == 1))[0]
    print("# testing samples = {}".format(len(idx_test)))
    x_test = x_test.toarray() / 255.0
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # params = {'l1_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0],
    #           'l2_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0]}
    params = {'l1_penalty': [0.0001, 0.001],
              'l2_penalty': [0.0001, 0.001]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="mnist_glm_logit_gridsearch",
              random_state=6789)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_glm_mnist_softmax_gridsearch():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    print("# training samples = {}".format(x_train.shape[0]))

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    print("# testing samples = {}".format(len(idx_test)))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    # params = {'l1_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0],
    #           'l2_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0]}
    params = {'l1_penalty': [0.0001, 0.001],
              'l2_penalty': [0.0001, 0.001]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = GLM(model_name="mnist_glm_softmax_gridsearch",
              link='softmax',
              loss='softmax',
              random_state=6789)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_glm_regression_gridsearch():
    # regression
    eps = 1e-6
    num_data = 100
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    # params = {'l1_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0],
    #           'l2_penalty': [0.0001, 0.001, 0.01, 0.1, 0.0]}
    params = {'l1_penalty': [0.0001, 0.001],
              'l2_penalty': [0.0001, 0.001]}

    ps = PredefinedSplit(test_fold=[-1] * 70 + [1] * 30)

    clf = GLM(model_name="glm_regression_gridsearch",
              task='regression',
              link='linear',  # link function
              loss='quadratic',  # loss function
              l2_penalty=0.0,  # ridge regularization
              l1_penalty=0.0,  # Lasso regularization
              l1_smooth=1E-5,  # smoothing for Lasso regularization
              l1_method='pseudo_huber',  # approximation method for L1-norm
              random_state=6789)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x[:70], y[:70])

    y_train_pred = best_clf.predict(x[:70])
    y_test_pred = best_clf.predict(x[70:])

    print("Training error = %.4f" % (metrics.mean_squared_error(y[:70], y_train_pred)))
    print("Testing error = %.4f" % (metrics.mean_squared_error(y[70:], y_test_pred)))


def test_glm_mnist_cv():
    from male import HOME
    x_train, y_train = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist_6k"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(HOME, "rdata/mnist/mnist.t_1k"),
                                        n_features=784)

    x_train = x_train.toarray() / 255.0
    idx_train = np.random.permutation(x_train.shape[0])
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]

    x_test = x_test.toarray() / 255.0
    idx_test = np.random.permutation(x_test.shape[0])
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]

    x = np.vstack([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    filepath = os.path.join(HOME, "rmodel/male/glm/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)
    loss_display = Display(title="Learning curves",
                           dpi='auto',
                           layout=(3, 1),
                           freq=4,
                           monitor=[{'metrics': ['loss', 'val_loss'],
                                     'type': 'line',
                                     'labels': ["training loss", "validation loss"],
                                     'title': "Learning losses",
                                     'xlabel': "epoch",
                                     'ylabel': "loss",
                                     },
                                    {'metrics': ['err', 'val_err'],
                                     'type': 'line',
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    {'metrics': ['err'],
                                     'type': 'line',
                                     'labels': ["training error"],
                                     'title': "Learning errors",
                                     'xlabel': "epoch",
                                     'ylabel': "error",
                                     },
                                    ])

    weight_display = Display(title="Filters",
                             dpi='auto',
                             layout=(1, 1),
                             figsize=(6, 15),
                             freq=10,
                             monitor=[{'metrics': ['weights'],
                                       'title': "Learned weights",
                                       'type': 'img',
                                       'tile_shape': (5, 2),
                                       },
                                      ])

    clf = GLM(model_name="mnist_glm_softmax",
              link='softmax',
              loss='softmax',
              optimizer='sgd',
              num_epochs=100,
              batch_size=100,
              learning_rate=0.001,
              task='classification',
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint, loss_display, weight_display],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    clf.fit(x, y)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_glm_check_grad()
    # test_glm_mnist_logit()
    # test_glm_mnist_softmax()
    # test_glm_mnist_logit_gridsearch()
    # test_glm_mnist_softmax_gridsearch()
    # test_glm_regression_gridsearch()
    # test_glm_mnist_cv()
