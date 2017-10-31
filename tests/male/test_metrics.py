from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest
import numpy as np

from male.configs import random_seed
from male.datasets import demo
from male.models.linear import GLM
from male.metrics import auc
from male.metrics import InceptionScore


def test_auc():
    print("========== Test AUC score ==========")

    (x_train, y_train), (x_test, y_test) = demo.load_pima()

    model = GLM(model_name="GLM_logit",
                l1_penalty=0.0,
                l2_penalty=0.0,
                random_state=random_seed())

    model.fit(x_train, y_train)
    train_auc = auc(y_train, model.predict(x_train))
    test_auc = auc(y_test, model.predict(x_test))
    print("Training AUC = %.4f" % train_auc)
    print("Testing AUC = %.4f" % test_auc)


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_inception_score():
    print("========== Test Inception score ==========")

    (x_train, y_train), (_, _) = demo.load_cifar10()
    x_train = x_train[:100].astype(np.float32).reshape([-1, 32, 32, 3]) * 255.0
    imgs = [0] * x_train.shape[0]
    for i in range(x_train.shape[0]):
        imgs[i] = x_train[i]
    score = InceptionScore.inception_score(imgs)
    print("Inception score: {:.4f}+-{:.4f}".format(score[0], score[1]))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_auc()
    # test_inception_score()
