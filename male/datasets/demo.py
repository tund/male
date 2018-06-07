from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pickle
import numpy as np

import networkx as nx
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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_tiny_shakespeare():
    # data I/O
    file_path = get_file("tinyshakespeare.txt", origin=remote_data_dir() + "/tinyshakespeare.txt",
                         cache_subdir="demo")

    data = open(file_path).read()
    # use set() to count the vacab size
    chars = list(set(data))

    # dictionary to convert char to idx, idx to char
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    return data, char_to_ix, ix_to_char


def load_text8_1pct():
    file_path = get_file("text8_1pct.txt",
                         origin=remote_data_dir() + "/text8_1pct.txt",
                         cache_subdir="demo")
    with open(file_path, "r") as f:
        txt = f.read()
    return txt


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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_synthetic_2d_semi(shuffle_data=True, randseed='default'):
    train_path = get_file("2d.semi.libsvm",
                          origin=remote_data_dir() + "/2d.semi.libsvm",
                          cache_subdir="demo")
    test_path = get_file("2d.semi.libsvm",
                         origin=remote_data_dir() + "/2d.semi.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=2)
    x_test, y_test = load_svmlight_file(test_path, n_features=2)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

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
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_20newsgroups(shuffle_data=True, randseed='default'):
    train_path = get_file("20news_bydate_5Kwordcount_in_entire_data_"
                          "countfeat_tiny_train.libsvm",
                          origin=remote_data_dir()
                                 + "/20news_bydate_5Kwordcount_in_entire_data_"
                                   "countfeat_tiny_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("20news_bydate_5Kwordcount_in_entire_data_"
                         "countfeat_tiny_test.libsvm",
                         origin=remote_data_dir()
                                + "/20news_bydate_5Kwordcount_in_entire_data_"
                                  "countfeat_tiny_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=5000)
    x_test, y_test = load_svmlight_file(test_path, n_features=5000)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist(shuffle_data=True, randseed='default'):
    train_path = get_file("fashion_mnist_train.libsvm",
                          origin=remote_data_dir() + "/fashion_mnist_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("fashion_mnist_test.libsvm",
                         origin=remote_data_dir() + "/fashion_mnist_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=784)
    x_test, y_test = load_svmlight_file(test_path, n_features=784)
    x_train = x_train.toarray() / 255.0
    x_test = x_test.toarray() / 255.0

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_a1a_semi(shuffle_data=True, randseed='default'):
    train_path = get_file("dla1a.txt_train.70u.libsvm",
                          origin=remote_data_dir() + "/dla1a.txt_train.70u.libsvm",
                          cache_subdir="demo")
    test_path = get_file("dla1a.txt_test.70u.libsvm",
                         origin=remote_data_dir() + "/dla1a.txt_test.70u.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, n_features=4)
    x_test, y_test = load_svmlight_file(test_path, n_features=4)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_yeast(shuffle_data=True, randseed='default'):
    train_path = get_file("yeast_train.libsvm", origin=remote_data_dir() + "/yeast_train.libsvm",
                          cache_subdir="demo")
    test_path = get_file("yeast_test.libsvm", origin=remote_data_dir() + "/yeast_test.libsvm",
                         cache_subdir="demo")

    x_train, y_train = load_svmlight_file(train_path, multilabel=True, n_features=103)
    x_test, y_test = load_svmlight_file(test_path, multilabel=True, n_features=103)
    x_train, y_train = x_train.toarray(), np.array(y_train)
    x_test, y_test = x_test.toarray(), np.array(y_test)

    if shuffle_data:
        x_train, y_train = shuffle(x_train, y_train, randseed=randseed)

    return (x_train, y_train), (x_test, y_test)


def load_wikipos():
    graph_path = get_file("wiki_pos.graph", origin=remote_data_dir() + "/wiki_pos.graph",
                          cache_subdir="demo")
    label_path = get_file("wiki_pos.labels", origin=remote_data_dir() + "/wiki_pos.labels",
                          cache_subdir="demo")
    walk_path = get_file("wiki_pos.walks", origin=remote_data_dir() + "/wiki_pos.walks",
                         cache_subdir="demo")

    G = nx.read_edgelist(graph_path, nodetype=int,
                         data=(('weight', float),),
                         create_using=nx.DiGraph()).to_undirected()
    labels = []
    with open(label_path) as fin:
        for line in fin:
            labels.append(tuple([int(i) for i in line.split()]))

    walks = []
    with open(walk_path) as fin:
        for line in fin:
            walks.append(list(map(int, line.split())))

    return G, np.array(labels), walks


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
