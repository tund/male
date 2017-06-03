from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest
import numpy as np

from male import TensorFlowModel
from male import random_seed
from male.datasets import demo
from male.models.linear import TensorFlowGLM


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_save_load():
    print("========== Test save, load tensorflow models ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    clf = TensorFlowGLM(model_name="iris_TensorFlowGLM_softmax",
                        link='softmax',
                        loss='softmax',
                        num_epochs=5,
                        random_state=random_seed())

    clf.fit(x_train, y_train)

    print("After training:")
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    save_file_path = clf.save()

    clf1 = TensorFlowModel.load_model(save_file_path)

    print("After save and load:")
    train_err1 = 1.0 - clf1.score(x_train, y_train)
    test_err1 = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err1)
    print("Testing error = %.4f" % test_err1)
    assert abs(train_err - train_err1) < 1e-6
    assert abs(test_err - test_err1) < 1e-6


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
def test_continue_training():
    print("========== Test continue training tensorflow models ==========")

    np.random.seed(random_seed())

    (x_train, y_train), (x_test, y_test) = demo.load_iris()
    print("Number of training samples = {}".format(x_train.shape[0]))
    print("Number of testing samples = {}".format(x_test.shape[0]))

    num_epochs = 5
    clf = TensorFlowGLM(model_name="iris_TensorFlowGLM_softmax",
                        link='softmax',
                        loss='softmax',
                        optimizer='sgd',
                        batch_size=10,
                        num_epochs=num_epochs,
                        random_state=random_seed())

    clf.fit(x_train, y_train)

    print("After training for {0:d} epochs".format(num_epochs))
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    clf.num_epochs = 10
    print("Set number of epoch to {0:d}, then continue training...".format(clf.num_epochs))
    clf.fit(x_train, y_train)
    train_err = 1.0 - clf.score(x_train, y_train)
    test_err = 1.0 - clf.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)

    save_file_path = clf.save()
    clf1 = TensorFlowModel.load_model(save_file_path)
    clf1.num_epochs = 15
    print("Save, load, set number of epoch to {0:d}, "
          "then continue training...".format(clf1.num_epochs))
    clf1.fit(x_train, y_train)
    train_err = 1.0 - clf1.score(x_train, y_train)
    test_err = 1.0 - clf1.score(x_test, y_test)
    print("Training error = %.4f" % train_err)
    print("Testing error = %.4f" % test_err)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_save_load()
    # test_continue_training()
