from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np

from male import random_seed
from male.datasets import demo
from male.optimizers import SGD
from male.optimizers import Adam
from male.optimizers import Nadam
from male.optimizers import Adamax
from male.optimizers import Adagrad
from male.optimizers import RMSProp
from male.optimizers import Adadelta
from male.models.linear import GLM
from male.optimizers.newton import Newton


def test_sgd_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using SGD with default parameters...")
    clf = GLM(model_name="sgd_glm",
              optimizer='sgd',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using SGD with customized parameters...")
    optz = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
    clf = GLM(model_name="sgd_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_adagrad_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using AdaGrad with default parameters...")
    clf = GLM(model_name="adagrad_glm",
              optimizer='adagrad',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using SGD with customized parameters...")
    optz = Adagrad(learning_rate=0.1)
    clf = GLM(model_name="adagrad_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_rmsprop_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using RMSProp with default parameters...")
    clf = GLM(model_name="rmsprop_glm",
              optimizer='rmsprop',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using RMSProp with customized parameters...")
    optz = RMSProp(learning_rate=0.01)
    clf = GLM(model_name="rmsprop_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_adadelta_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using AdaDelta with default parameters...")
    clf = GLM(model_name="adadelta_glm",
              optimizer='adadelta',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using AdaDelta with customized parameters...")
    optz = Adadelta(learning_rate=0.1)
    clf = GLM(model_name="adadelta_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_adam_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using Adam with default parameters...")
    clf = GLM(model_name="adam_glm",
              optimizer='adam',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using Adam with customized parameters...")
    optz = Adam(learning_rate=0.002)
    clf = GLM(model_name="adam_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_adamax_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using AdaMax with default parameters...")
    clf = GLM(model_name="adamax_glm",
              optimizer='adamax',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using AdaMax with customized parameters...")
    optz = Adamax(learning_rate=0.004)
    clf = GLM(model_name="adamax_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def test_nadam_glm():
    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    print("Training GLM using NaDam with default parameters...")
    clf = GLM(model_name="nadam_glm",
              optimizer='nadam',
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    print("Training GLM using NaDam with customized parameters...")
    optz = Nadam(learning_rate=0.004)
    clf = GLM(model_name="nadam_glm",
              optimizer=optz,
              num_epochs=10,
              link='softmax',
              loss='softmax',
              random_state=random_seed())

    clf.fit(x_train, y_train)

    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


def obj_func_f1(x, c):
    """
    f1 = x^2+y^2-3x+6y
    min = -11.25 at x = 1.5, y = -3
    """
    a, b = c
    return x[0] ** 2 + (x[1] ** 2) + a * x[0] + b * x[1]


def grad_func_f1(x, c):
    a, b = c
    return np.array([2 * x[0] + a, 2 * x[1] + b])


def hess_func_f1(x, c):
    a, b = c
    return np.diag(np.array([2, 2]))


def test_newton():
    opt = Newton(learning_rate=0.8, tolerance=1e-5, max_loop=10)
    opt.init_params(
        obj_func=obj_func_f1,
        grad_func=grad_func_f1,
        hess_func=hess_func_f1
    )
    x = np.array([10.0, 10.0])
    a = -3
    b = 6
    x, obj = opt.solve(x, (a, b))
    assert (x[0] == 1.5) and (x[1] == -3.0) and (obj == -11.25)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_sgd_glm()
    # test_adagrad_glm()
    # test_rmsprop_glm()
    # test_adadelta_glm()
    # test_adam_glm()
    # test_adamax_glm()
    # test_nadam_glm()
    # test_newton()
