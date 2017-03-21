from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from sklearn.datasets import load_svmlight_file

from .. import remote_data_dir
from ..utils.data_utils import get_file


def load_mnist():
    train_path = get_file("mnist_train", origin=remote_data_dir() + "/mnist_train",
                          cache_subdir="demo")
    test_path = get_file("mnist_test", origin=remote_data_dir() + "/mnist_test",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=784)
    x_test, y_test = load_svmlight_file(test_path, n_features=784)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    return (x_train, y_train), (x_test, y_test)


def load_svmguide1():
    train_path = get_file("svmguide1_train", origin=remote_data_dir() + "/svmguide1_train",
                          cache_subdir="demo")
    test_path = get_file("svmguide1_test", origin=remote_data_dir() + "/svmguide1_test",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=4)
    x_test, y_test = load_svmlight_file(test_path, n_features=4)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    return (x_train, y_train), (x_test, y_test)


def load_synthetic_2d():
    train_path = get_file("synthetic_2D_data_train",
                          origin=remote_data_dir() + "/synthetic_2D_data_train",
                          cache_subdir="demo")
    test_path = get_file("synthetic_2D_data_test",
                         origin=remote_data_dir() + "/synthetic_2D_data_test",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=2)
    x_test, y_test = load_svmlight_file(test_path, n_features=2)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    return (x_train, y_train), (x_test, y_test)
