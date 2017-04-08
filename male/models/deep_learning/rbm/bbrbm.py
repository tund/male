from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from scipy.misc import logsumexp
from sklearn.utils.validation import check_is_fitted
from .rbm import RBM
from ....utils.func_utils import sigmoid, logsumone

EPSILON = np.finfo(np.float32).eps


class BernoulliBernoulliRBM(RBM):
    def __init__(self, model_name="BBRBM", **kwargs):
        super(BernoulliBernoulliRBM, self).__init__(model_name=model_name, **kwargs)

    def _init(self):
        super(BernoulliBernoulliRBM, self)._init()

    def _init_params(self, x):
        super(BernoulliBernoulliRBM, self)._init_params(x)
        self.pcd_hsample = (0.5 > np.random.rand(self.num_chains, self.num_hidden)).astype(np.int)

    def _get_hidden_prob(self, vsample, **kwargs):
        return sigmoid(vsample.dot(self.w) + self.h)

    def _sample_hidden(self, hprob):
        return (hprob > np.random.rand(*hprob.shape)).astype(np.int)

    def _get_visible_prob(self, hsample):
        return sigmoid(hsample.dot(self.w.T) + self.v)

    def _sample_visible(self, vprob):
        return (vprob > np.random.rand(*vprob.shape)).astype(np.int)

    def get_free_energy(self, x):
        wx = x.dot(self.w) + self.h
        return - x.dot(self.v.T) - logsumone(wx).sum(axis=1, keepdims=True)
        # return - x.dot(self.v_.T) \
        #        - np.logaddexp(np.zeros(wx.shape), wx).sum(axis=1, keepdims=True)

    def get_conditional_loglik(self, x, hsample):
        vprob = np.clip(self._get_visible_prob(hsample), EPSILON, 1 - EPSILON)
        return logsumexp(x.dot(np.log(vprob.T)) + (1 - x).dot(np.log(1 - vprob.T)), axis=1) \
               - np.log(hsample.shape[0])

    def get_reconstruction_loglik(self, x, rdata=None):
        rdata = self.get_reconstruction(x) if rdata is None else rdata
        rdata = np.clip(rdata, EPSILON, 1 - EPSILON)
        return x * np.log(rdata) + (1 - x) * np.log(1 - rdata)

    def get_logpartition(self, method='exact'):
        if method == 'exact':
            if min(self.num_hidden, self.num_visible) > 20:
                raise ValueError("The BernoulliBernoulliRBM is too big to "
                                 "compute log-partition function exactly.")
            if self.num_hidden < self.num_visible:
                hsample = np.zeros((2 ** self.num_hidden, self.num_hidden))
                for i in range(2 ** self.num_hidden):
                    hsample[i] = [int(j) for j in list("{0:b}".format(i).zfill(self.num_hidden))]
                log_hprob = (hsample.dot(self.h.T)
                             + logsumone(self.v + hsample.dot(self.w.T)).sum(axis=1, keepdims=True))
                return logsumexp(log_hprob)
            else:
                vsample = np.zeros((2 ** self.num_visible, self.num_visible))
                for i in range(2 ** self.num_visible):
                    vsample[i] = [int(j) for j in list("{0:b}".format(i).zfill(self.num_visible))]
                log_vprob = (vsample.dot(self.v.T)
                             + logsumone(self.h + vsample.dot(self.w)).sum(axis=1, keepdims=True))
                return logsumexp(log_vprob)
        else:
            raise NotImplementedError

    def transform(self, x):
        check_is_fitted(self, "w")
        return self._get_hidden_prob(x)

    def get_params(self, deep=True):
        out = super(BernoulliBernoulliRBM, self).get_params(deep=deep)
        param_names = BernoulliBernoulliRBM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
