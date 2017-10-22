from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pickle
import numpy as np
from sklearn.datasets import load_svmlight_file
from ..configs import random_seed
from ..configs import remote_data_dir
from ..utils.data_utils import get_file


def load_mnist(shuffle_data=True, randseed='default'):
    train_path = get_file("mnist_train.libsvm", origin=remote_data_dir() + "/mnist_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("mnist_test.libsvm", origin=remote_data_dir() + "/mnist_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=784)
    x_test, y_test = load_svmlight_file(test_path, n_features=784)
    x_train = x_train.toarray() / 255.0
    x_test = x_test.toarray() / 255.0

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_cifar10(shuffle_data=True, randseed='default'):
    train_path = get_file("cifar10_5k_train.pkl",
                          origin=remote_data_dir() + "/cifar10_5k_train.pkl",
                          cache_subdir="demo")
    test_path = get_file("cifar10_1k_test.pkl",
                         origin=remote_data_dir() + "/cifar10_1k_test.pkl",
                         cache_subdir="demo")

    tmp = pickle.load(open(train_path, "rb"))
    x_train, y_train = tmp['data'] / 255.0, tmp['labels']
    tmp = pickle.load(open(test_path, "rb"))
    x_test, y_test = tmp['data'] / 255.0, tmp['labels']

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_svmguide1(shuffle_data=True, randseed='default'):
    train_path = get_file("svmguide1_train.libsvm",
                          origin=remote_data_dir() + "/svmguide1_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("svmguide1_test.libsvm",
                         origin=remote_data_dir() + "/svmguide1_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=4)
    x_test, y_test = load_svmlight_file(test_path, n_features=4)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_iris(shuffle_data=True, randseed='default'):
    train_path = get_file("iris_scale_train.libsvm",
                          origin=remote_data_dir() + "/iris_scale_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("iris_scale_test.libsvm",
                         origin=remote_data_dir() + "/iris_scale_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=4)
    x_test, y_test = load_svmlight_file(test_path, n_features=4)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_pima(shuffle_data=True, randseed='default'):
    train_path = get_file("pima_train.libsvm",
                          origin=remote_data_dir() + "/pima_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("pima_test.libsvm",
                         origin=remote_data_dir() + "/pima_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=8)
    x_test, y_test = load_svmlight_file(test_path, n_features=8)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_synthetic_2d(shuffle_data=True, randseed='default'):
    train_path = get_file("synthetic_2D_data_train.libsvm",
                          origin=remote_data_dir() + "/synthetic_2D_data_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("synthetic_2D_data_test.libsvm",
                         origin=remote_data_dir() + "/synthetic_2D_data_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=2)
    x_test, y_test = load_svmlight_file(test_path, n_features=2)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_housing(shuffle_data=True, randseed='default'):
    train_path = get_file("housing_scale_train.libsvm",
                          origin=remote_data_dir() + "/housing_scale_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("housing_scale_test.libsvm",
                         origin=remote_data_dir() + "/housing_scale_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=13)
    x_test, y_test = load_svmlight_file(test_path, n_features=13)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_20newsgroups(shuffle_data=True, randseed='default'):
    train_path = get_file("20news_bydate_5Kwordcount_in_entire_data_countfeat_tiny_train.libsvm",
                          origin=remote_data_dir()
                                 + "/20news_bydate_5Kwordcount_in_entire_data_countfeat_tiny_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("20news_bydate_5Kwordcount_in_entire_data_countfeat_tiny_test.libsvm",
                         origin=remote_data_dir()
                                + "/20news_bydate_5Kwordcount_in_entire_data_countfeat_tiny_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=5000)
    x_test, y_test = load_svmlight_file(test_path, n_features=5000)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist(shuffle_data=True, randseed='default'):
    train_path = get_file("fashion_mnist_train.libsvm", origin=remote_data_dir() + "/fashion_mnist_test.libsvm",
                          cache_subdir="demo")
    test_path = get_file("fashion_mnist_test.libsvm", origin=remote_data_dir() + "/fashion_mnist_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=784)
    x_test, y_test = load_svmlight_file(test_path, n_features=784)
    x_train = x_train.toarray() / 255.0
    x_test = x_test.toarray() / 255.0

    if shuffle_data:
        shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def shuffle(x, y=None, randseed='default'):
    if randseed == 'default':
        np.random.seed(random_seed())
    elif isinstance(randseed, int):
        np.random.seed(randseed)
    idx_train = np.random.permutation(x.shape[0])
    x = x[idx_train]
    if y is not None:
        y = y[idx_train]
    return x, y
