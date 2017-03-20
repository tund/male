from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn.datasets import load_svmlight_file

from male import data_dir
from male.models.deep_learning.autoencoder import KerasVAE


def test_keras_vae_mnist_bin():
    x_train, y_train = load_svmlight_file(os.path.join(data_dir(), "demo/mnist_train"),
                                          n_features=784)
    x_test, y_test = load_svmlight_file(os.path.join(data_dir(), "demo/mnist_test"),
                                        n_features=784)
    x_train = x_train.toarray().astype(np.float32) / 255.0
    x_test = x_test.toarray().astype(np.float32) / 255.0

    cv = [-1] * x_train.shape[0] + [0] * x_test.shape[0]

    x = np.vstack((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    model = KerasVAE(num_visible=784,
                     num_hiddens=[500],
                     act_funcs=['sigmoid'],
                     cv=cv,
                     num_z=200,
                     z_init=1.0,
                     batch_size=128,
                     num_epochs=1000)
    model.fit(x)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_keras_vae_mnist_bin()
