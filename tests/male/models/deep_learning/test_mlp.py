from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from male.configs import random_seed
from male.datasets import demo
from male.models.deep_learning import MLP


def test_mlp_check_grad():
    eps = 1e-6
    num_data = 10
    num_features = 4
    num_classes = 3
    x = np.random.rand(num_data, num_features)
    y = np.random.randint(0, num_classes, num_data)

    model = MLP(model_name="checkgrad_MLP_softmax",
                task='classification',
                hidden_units_list=(5,),
                reg_lambda=0.0,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = MLP(model_name="checkgrad_MLP_softmax",
                task='classification',
                hidden_units_list=(5,),
                reg_lambda=0.01,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps

    model = MLP(model_name="checkgrad_MLP_softmax",
                task='classification',
                hidden_units_list=(5,),
                reg_lambda=0.1,
                random_state=random_seed())
    assert model.check_grad(x, y) < eps


def test_mlp_softmax():
    print("========== Test MLP for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()

    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = MLP(model_name="MLP_softmax",
              hidden_units_list=(5,),
              batch_size=16,
              num_epochs=4,
              learning_rate=0.1,
              reg_lambda=0.01,
              random_state=random_seed())

    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_mlp_softmax_gridsearch():
    print("========== Tune parameters for MLP for multiclass classification ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    params = {'learning_rate': [0.1, 0.05, 0.01],
              'hidden_units_list': [(1,), (5,), (20,)],
              'reg_lambda': [0.01, 0.001, 0.0001]}

    ps = PredefinedSplit(test_fold=[-1] * x_train.shape[0] + [1] * x_test.shape[0])

    clf = MLP(model_name="mlp_softmax_gridsearch",
              num_epochs=4,
              catch_exception=True,
              random_state=random_seed())

    gs = GridSearchCV(clf, params, cv=ps, n_jobs=-1, refit=False, verbose=True)
    gs.fit(x, y)

    print("Best error {} @ params {}".format(1 - gs.best_score_, gs.best_params_))

    best_clf = clone(clf).set_params(**gs.best_params_)
    best_clf.fit(x_train, y_train)

    train_err = 1.0 - best_clf.score(x_train, y_train)
    test_err = 1.0 - best_clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)
    assert abs(test_err - (1.0 - gs.best_score_)) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__])
    # test_mlp_check_grad()
    # test_mlp_softmax()
    # test_mlp_softmax_gridsearch()
