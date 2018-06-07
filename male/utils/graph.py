from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed


from ..models.distributions import Multinomial


class Graph(object):
    """Adapted from [Aditya Grover]
    """

    def __init__(self, nx_G, is_directed=True,
                 p=1.0, q=1.0,
                 num_workers=4, random_state=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.num_workers = num_workers
        self.random_state = random_state
        self.random_engine = check_random_state(self.random_state)

    def get_unbiased_randomwalk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.G.neighbors(cur)
            if len(cur_nbrs) > 0:
                walk.append(self.random_engine.choice(cur_nbrs))
            else:
                break
        return walk

    def get_biased_randomwalk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.G.neighbors(cur)
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_nodes[cur].sample(1)[0]])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[self.alias_edges[(prev, cur)].sample(1)[0]]
                    walk.append(next)
            else:
                break
        return walk

    def get_random_walks(self, num_walks, walk_length, biased=False):
        '''
        Repeatedly simulate random walks from each node.
        '''
        random_walk = self.get_biased_randomwalk if biased == True else self.get_unbiased_randomwalk
        if self.num_workers == 1:
            walks = []
            nodes = list(self.G.nodes())
            for walk_iter in range(num_walks):
                self.random_engine.shuffle(nodes)
                for node in nodes:
                    walks.append(random_walk(walk_length, node))
        else:
            nodes = list(self.G.nodes()) * num_walks
            self.random_engine.shuffle(nodes)
            walk_len = [walk_length] * len(nodes)
            walks = Parallel(n_jobs=self.num_workers)(delayed(random_walk)(i, j)
                                                      for i, j in zip(walk_len, nodes))
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        nbrs = self.G.neighbors(dst)
        unnormalized_probs = np.zeros(len(nbrs))
        for i, dst_nbr in enumerate(nbrs):
            if dst_nbr == src:
                unnormalized_probs[i] = self.G[dst][dst_nbr]['weight'] / self.p
            elif self.G.has_edge(dst_nbr, src):
                unnormalized_probs[i] = self.G[dst][dst_nbr]['weight']
            else:
                unnormalized_probs[i] = self.G[dst][dst_nbr]['weight'] / self.q

        return Multinomial(probs=unnormalized_probs / np.sum(unnormalized_probs))

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        is_directed = self.is_directed

        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = np.array([self.G[node][nbr]['weight']
                                           for nbr in self.G.neighbors(node)])
            alias_nodes[node] = Multinomial(probs=unnormalized_probs / np.sum(unnormalized_probs))

        alias_edges = {}

        if is_directed:
            for edge in self.G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in self.G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
