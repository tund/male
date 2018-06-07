from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
from gensim.models import Word2Vec
from sklearn.utils.validation import check_is_fitted

from male import Model
from male.utils.graph import Graph


class Node2Vec(Model):

    def __init__(self,
                 model_name='Node2Vec',
                 emb_size=128,
                 window=10,
                 num_walks=10,
                 walk_length=80,
                 p=1.0, q=1.0,
                 num_neg_samples=5,
                 num_workers=4,
                 directed=True,
                 **kwargs
                 ):
        super(Node2Vec, self).__init__(model_name=model_name, **kwargs)
        self.emb_size = emb_size
        self.window = window
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_neg_samples = num_neg_samples
        self.num_workers = num_workers
        self.directed = directed

    def _init(self):
        super(Node2Vec, self)._init()
        self.G = None

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None, **kwargs):
        if 'walks' in kwargs:
            self.walks = kwargs['walks']
        else:
            if 'graph' in kwargs:
                self._set_graph(kwargs['graph'])
            if self.G is None:
                sys.exit("Graph has not been initialized.")
            self.walks = self.G.get_random_walks(self.num_walks, self.walk_length, biased=True)

        self.walks = [list(map(str, walk)) for walk in self.walks]
        # Learn embeddings by optimizing the Skipgram objective using SGD.
        self.w2v = Word2Vec(self.walks, size=self.emb_size, window=self.window, min_count=0,
                            negative=self.num_neg_samples, sg=1, seed=self.random_state,
                            workers=self.num_workers, iter=self.num_epochs, compute_loss=True)

    def _set_graph(self, graph):
        self.G = Graph(graph, self.directed, self.p, self.q)
        self.G.preprocess_transition_probs()

    def transform(self, x):
        check_is_fitted(self, "w2v")
        return self.w2v[list(map(str, x))]
