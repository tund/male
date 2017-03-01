from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import numpy as np
import copy

from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from scipy.misc import logsumexp
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .bbrbm import BernoulliBernoulliRBM
from .rbm import INFERENCE_ENGINE, LEARNING_METHOD, CD_SAMPLING
from ....utils.generic_utils import make_batches
from ....utils.func_utils import sigmoid, softmax

EPSILON = np.finfo(np.float32).eps
NONNEGATIVE_TYPE = {'barrier': 0, 'force': 1, 'square': 2, 'exp': 3}


class NonnegativeRBM(BernoulliBernoulliRBM):
    """ Nonnegative Restricted Boltzmann Machines
    """

    def __init__(self,
                 model_name="NRBM",
                 nonnegative_type='barrier',
                 nonnegative_cost=0.1,
                 *args, **kwargs):
        kwargs["model_name"] = model_name
        super(NonnegativeRBM, self).__init__(**kwargs)
        self.nonnegative_type = nonnegative_type
        self.nonnegative_cost = nonnegative_cost

    def _init(self):
        super(NonnegativeRBM, self)._init()
        try:
            self.nonnegative_type = NONNEGATIVE_TYPE[self.nonnegative_type]
        except KeyError:
            raise ValueError("Nonnegative type %s is not supported." % self.nonnegative_type)

    def _init_params(self, x):
        super(NonnegativeRBM, self)._init_params(x)
        # initialize parameters
        k, n = self.num_hidden, self.num_visible
        self.w_ = self.w_init * np.random.rand(n, k)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        k, n = self.num_hidden, self.num_visible
        prev_hprob = np.zeros([1, k])

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch_ < self.num_epochs) and (not self.stop_training_):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch_)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]

                pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

                # ======= clamp phase ========
                hprob = self._get_hidden_prob(x_batch)

                # sparsity
                if self.sparse_weight > 0:
                    hg, wg, prev_hprob = self._hidden_sparsity(x_batch, prev_hprob, hprob)
                    pos_hgrad += hg
                    pos_wgrad += wg

                hg, vg, wg = self._get_positive_grad(x_batch, hprob)
                pos_hgrad += hg
                pos_vgrad += vg
                pos_wgrad += wg

                # ======== free phase =========
                if self.learning_method == LEARNING_METHOD['cd']:
                    for icd in range(self.num_cd - 1):
                        hsample, vprob, vsample, hprob = self._gibbs_sampling(
                            hprob, sampling=CD_SAMPLING['hidden_visible'])
                    hsample, vprob, vsample, hprob = self._gibbs_sampling(
                        hprob, sampling=self.sampling_in_last_cd)

                # ======== negative phase =========
                neg_hgrad, neg_vgrad, neg_wgrad = self._get_negative_grad(vprob, hprob)

                # update params
                self.hgrad_inc_ = self.momentum_ * self.hgrad_inc_ \
                                  + self.learning_rate * (pos_hgrad - neg_hgrad)
                self.vgrad_inc_ = self.momentum_ * self.vgrad_inc_ \
                                  + self.learning_rate * (pos_vgrad - neg_vgrad)
                self.wgrad_inc_ = self.momentum_ * self.wgrad_inc_ \
                                  + self.learning_rate * (pos_wgrad - neg_wgrad
                                                          - self.weight_cost * self.w_)

                # nonnegative update
                if self.nonnegative_type == NONNEGATIVE_TYPE['barrier']:
                    idx = self.w_ < 0
                    self.wgrad_inc_[idx] -= self.learning_rate \
                                            * self.nonnegative_cost * self.w_[idx]

                self.h_ += self.hgrad_inc_
                self.v_ += self.vgrad_inc_
                self.w_ += self.wgrad_inc_

                # nonnegative update
                if self.nonnegative_type == NONNEGATIVE_TYPE['force']:
                    self.w_[self.w_ < 0] = 0

                batch_logs.update(self._on_batch_end(x_batch, rdata=vprob))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

    def get_params(self, deep=True):
        out = super(NonnegativeRBM, self).get_params(deep=deep)
        out.update({'nonnegative_type': self.nonnegative_type,
                    'nonnegative_cost': self.nonnegative_cost})
        return out
