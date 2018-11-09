from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest
import numpy as np
from sklearn.model_selection import ShuffleSplit

from male.configs import random_seed
from male.datasets import demo
from male.models.linear import GLM
from male.models.embedding import Node2Vec


def test_node2vec_wiki_pos(show_figure=False, block_figure_on_end=False):
    graph, labels, walks = demo.load_wikipos()
    y = labels
    x = np.array(range(len(labels)))
    model = Node2Vec(
        model_name='Node2Vec',
        emb_size=16,
        window=3,
        num_walks=2,
        walk_length=10,
        p=1.0, q=1.0,
        num_workers=4,
        directed=False,
    )
    model.fit(x, y, walks=walks)

    train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.1,
                                            random_state=random_seed()).split(x))
    x_train, y_train = model.transform(x[train_idx]), y[train_idx]
    x_test, y_test = model.transform(x[test_idx]), y[test_idx]

    clf = GLM(model_name="GLM_multilogit",
              task='multilabel',
              link='logit',
              loss='multilogit',
              random_state=random_seed())

    clf.fit(x_train, y_train)
    train_f1 = clf.score(x_train, y_train)
    test_f1 = clf.score(x_test, y_test)
    print("Training weighted-F1-macro = %.4f" % train_f1)
    print("Testing weighted-F1-macro = %.4f" % test_f1)


if __name__ == '__main__':
    pytest.main([__file__])
    # test_node2vec_wiki_pos(show_figure=True, block_figure_on_end=True)
