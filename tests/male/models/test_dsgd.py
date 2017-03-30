from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pytest
import numpy as np

from sklearn.datasets import load_svmlight_file

from male import data_dir
from male import random_seed
from male.models.kernel import DualSGD


def test_dualsgd_codrna():
    np.random.seed(random_seed())
    # x in [-1, 1]
    x_train, y_train = load_svmlight_file(
        os.path.join(data_dir(), "cod-rna/cod-rna.scale_1k"), n_features=8)
    x_train = x_train.toarray()
    idx = np.random.permutation(x_train.shape[0])

    est = DualSGD(model_name="codrna_dualsgd_logit",
                  k=20,
                  D=200,
                  gamma=1.0,
                  lbd=3.3593684387335183e-05,
                  loss='hinge',
                  maintain='k-merging',
                  max_budget_size=100)
    est.fit(x_train[idx], y_train[idx])
    print("Mistake rate = %.4f" % est.last_score_)
    print("Training time = %.4f" % est.train_time_)
    print("Budget size = %d" % est.budget_size_)


def test_dualsgd_airlines():
    np.random.seed(random_seed())
    # x in [-1, 1]
    x_train, y_train = load_svmlight_file(
        os.path.join(data_dir(), "airlines/2008_depdelay_minute_8feats_standardize01.libsvm_100k"),
        n_features=8)
    x_train = x_train.toarray()
    idx = np.random.permutation(x_train.shape[0])

    est = DualSGD(model_name="airlines_dualsgd_eps_insensitive",
                  k=20,
                  D=200,
                  gamma=1.0,
                  eps=0.001,
                  lbd=0.00128,
                  loss='eps_insensitive',
                  maintain='k-merging',
                  max_budget_size=100,
                  random_state=6789)
    est.fit(x_train[idx], y_train[idx])
    print("RMSE = %.4f" % np.sqrt(est.last_score_))
    print("Training time = %.4f" % est.train_time_)
    print("Budget size = %d" % est.budget_size_)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_dualsgd_codrna()
    # test_dualsgd_airlines()
