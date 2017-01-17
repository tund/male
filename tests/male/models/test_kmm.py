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

from male.models.kernel import KMM
from male.callbacks import EarlyStopping
from male.callbacks import ModelCheckpoint


def test_kmm_check_grad():
    # <editor-fold desc="Binary classification using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_kmm_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Binary classification using MULTIPLE kernels">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 2
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_kmm_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>s

    # <editor-fold desc="Multiclass classification using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_kmm_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Multiclass classification using MULTIPLE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_classes = 3
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = KMM(model_name="checkgrad_kmm_hinge",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='hinge',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_logit",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='logit',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)
    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression using ONE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_kernels = 1
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = KMM(model_name="checkgrad_kmm_l1",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l1',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_l2",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l2',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>

    # <editor-fold desc="Regression using MULTIPLE kernel">
    eps = 1e-6
    num_data = 10
    num_features = 5
    num_kernels = 2
    x = np.random.rand(num_data, num_features)
    y = np.random.randn(num_data)
    model = KMM(model_name="checkgrad_kmm_l1",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l1',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_l2",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='l2',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps

    model = KMM(model_name="checkgrad_kmm_eps",
                D=4,
                lbd=0.01,
                gamma=0.125,
                mode='batch',
                loss='eps_insensitive',
                temperature=0.1,
                num_kernels=num_kernels,
                learning_rate=0.001,
                learning_rate_mu=0.001,
                learning_rate_gamma=0.001,
                learning_rate_alpha=0.001)

    assert model.check_grad(x, y) < eps
    assert model.check_grad_online(x, y) < eps
    # </editor-fold>


def test_kmm_mnist_bin():
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

    clf = KMM(model_name="mnist_kmm_hinge",
              D=4,
              lbd=0.01,
              gamma=0.125,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              temperature=0.1,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = KMM(model_name="mnist_kmm_hinge",
              D=4,
              lbd=0.01,
              gamma=0.125,
              mode='online',
              loss='hinge',
              num_kernels=4,
              temperature=0.1,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_rate_)


def test_kmm_mnist_softmax():
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

    clf = KMM(model_name="mnist_kmm_hinge",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))

    clf = KMM(model_name="mnist_kmm_hinge",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='online',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    clf.fit(x_train, y_train)

    print("Mistake rate = %.4f" % clf.mistake_rate_)


def test_kmm_mnist_softmax_gridsearch():
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

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = KMM(model_name="mnist_kmm_hinge",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_kmm_regression_gridsearch():
    # regression
    eps = 1e-6
    num_data = 100
    num_features = 5
    x = np.random.rand(num_data, num_features)
    y = np.random.rand(num_data)

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'learning_rate': [0.0001, 0.001, 0.003, 0.01, 0.3, 0.1]}

    ps = PredefinedSplit(test_fold=[-1] * 70 + [1] * 30)

    clf = KMM(model_name="mnist_kmm_l2",
              D=4,
              lbd=0.01,
              gamma=0.01,
              mode='batch',
              loss='l2',
              num_kernels=4,
              batch_size=100,
              temperature=0.1,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.001,
              learning_rate_mu=0.001,
              learning_rate_gamma=0.001,
              learning_rate_alpha=0.001,
              random_state=6789,
              verbose=1)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(-gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x[:70], y[:70])

    y_train_pred = best_clf.predict(x[:70])
    y_test_pred = best_clf.predict(x[70:])

    print("Training error = %.4f" % (metrics.mean_squared_error(y[:70], y_train_pred)))
    print("Testing error = %.4f" % (metrics.mean_squared_error(y[70:], y_test_pred)))


def test_kmm_mnist_cv():
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

    early_stopping = EarlyStopping(monitor='val_err', patience=10)
    filepath = os.path.join(HOME, "rmodel/male/kmm/mnist_{epoch:04d}_{val_err:.6f}.pkl")
    checkpoint = ModelCheckpoint(filepath,
                                 mode='min',
                                 monitor='val_err',
                                 verbose=0,
                                 save_best_only=True)

    # <editor-fold desc="Best params">
    # clf = KMM(model_name="mnist_kmm_hinge",
    #           D=200,
    #           lbd=0.0,
    #           gamma=0.1,
    #           mode='batch',
    #           loss='hinge',
    #           num_kernels=10,
    #           batch_size=100,
    #           temperature=1.0,
    #           num_epochs=50,
    #           num_nested_epochs=1,
    #           learning_rate=0.001,
    #           learning_rate_mu=0.0,
    #           learning_rate_gamma=0.001,
    #           learning_rate_alpha=0.001,
    #           metrics=['loss', 'err'],
    #           callbacks=[early_stopping, checkpoint],
    #           cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
    #           random_state=6789,
    #           verbose=1)
    # </editor-fold>

    clf = KMM(model_name="mnist_kmm_hinge",
              D=20,
              lbd=0.0,
              gamma=0.1,
              mode='batch',
              loss='hinge',
              num_kernels=3,
              batch_size=100,
              temperature=1.0,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.1,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.1,
              learning_rate_alpha=0.1,
              metrics=['loss', 'err'],
              callbacks=[early_stopping, checkpoint],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    clf.fit(x, y)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


def test_kmm_mnist_cv_gridsearch():
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

    x = np.vstack([x_train, x_test, x_test])
    y = np.concatenate([y_train, y_test, y_test])

    params = {'gamma': [0.1, 0.125, 0.5, 1.0],
              'temperature': [0.1, 0.5, 1.0],
              'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0]
                                   + [-1] * x_test.shape[0]
                                   + [1] * x_test.shape[0])

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    # filepath = os.path.join(HOME, "rmodel/male/kmm/mnist_{epoch:04d}_{val_loss:.6f}.pkl")
    # checkpoint = ModelCheckpoint(filepath,
    #                              mode='min',
    #                              monitor='val_loss',
    #                              verbose=0,
    #                              save_best_only=True)

    clf = KMM(model_name="mnist_kmm_hinge",
              D=20,
              lbd=0.0,
              gamma=0.1,
              mode='batch',
              loss='hinge',
              num_kernels=4,
              batch_size=100,
              temperature=1.0,
              num_epochs=50,
              num_nested_epochs=1,
              learning_rate=0.1,
              learning_rate_mu=0.0,
              learning_rate_gamma=0.1,
              learning_rate_alpha=0.1,
              metrics=['loss', 'err'],
              # callbacks=[early_stopping, checkpoint],
              callbacks=[early_stopping],
              cv=[-1] * x_train.shape[0] + [0] * x_test.shape[0],
              random_state=6789,
              verbose=1)

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best score {} @ params {}".format(gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)

    best_clf.fit(np.vstack([x_train, x_test]),
                 np.concatenate([y_train, y_test]))

    y_train_pred = best_clf.predict(x_train)
    y_test_pred = best_clf.predict(x_test)

    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing error = %.4f" % (1 - metrics.accuracy_score(y_test, y_test_pred)))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_kmm_check_grad()
    # test_kmm_mnist_bin()
    # test_kmm_mnist_softmax()
    # test_kmm_mnist_softmax_gridsearch()
    # test_kmm_regression_gridsearch()
    # test_kmm_mnist_cv()
    # test_kmm_mnist_cv_gridsearch()
