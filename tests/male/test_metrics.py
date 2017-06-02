from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pytest
import numpy as np

from male.datasets import demo
from male.metrics import init_inception, inception_score


@pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires tensorflow library")
@pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
def test_inception_score():
    (x_train, y_train), (_, _) = demo.load_cifar10()
    x_train = x_train[:100].astype(np.float32).reshape([-1, 32, 32, 3]) * 255.0
    imgs = [0] * x_train.shape[0]
    for i in range(x_train.shape[0]):
        imgs[i] = x_train[i]
    inception_model, inception_graph = init_inception()
    score = inception_score(inception_model, inception_graph, imgs)
    print("Inception score: {:.4f}+-{:.4f}".format(score[0], score[1]))


if __name__ == '__main__':
    pytest.main([__file__])
    # test_inception_score()
