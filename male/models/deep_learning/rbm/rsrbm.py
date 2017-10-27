from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from scipy.misc import logsumexp
from sklearn.utils.validation import check_is_fitted
from .rbm import RBM
from .rbm import CD_SAMPLING
from .rbm import LEARNING_METHOD
from ....utils.func_utils import sigmoid, logsumone
from ....utils.generic_utils import make_batches

EPSILON = np.finfo(np.float32).eps


class ReplicatedSoftmaxRBM(RBM):
    def __init__(self,
                 model_name="RSRBM",
                 learning_rate=0.01,
                 learning_rate_hidden=0.0001,
                 sampling_in_last_cd="visible",
                 **kwargs):
        super(ReplicatedSoftmaxRBM, self).__init__(model_name=model_name,
                                                   learning_rate=learning_rate,
                                                   sampling_in_last_cd=sampling_in_last_cd,
                                                   **kwargs)
        self.learning_rate_hidden = learning_rate_hidden

    def _init(self):
        super(ReplicatedSoftmaxRBM, self)._init()

    def _init_params(self, x):
        super(ReplicatedSoftmaxRBM, self)._init_params(x)
        self.pcd_hsample = (0.5 > np.random.rand(self.num_chains, self.num_hidden)).astype(np.int)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        """Fit the model to the data X

        Parameters
        ----------
        x : {array-like, sparse matrix} shape (num_samples, num_visible)
            Training data.

        Returns
        -------
        self : RBM
            The fitted model.
        """

        # transform data into log-space
        x = np.round(np.log(x + 1))
        if x_valid is not None:
            x_valid = np.round(np.log(x_valid + 1))

        k, n = self.num_hidden, self.num_visible
        prev_hprob = np.zeros([1, k])

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]

                pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

                # ======= clamp phase ========
                data_len = np.sum(x_batch, axis=1, keepdims=True)
                hprob = self._get_hidden_prob(x_batch, data_len=data_len)

                # sparsity
                if self.sparse_weight > 0:
                    hg, wg, prev_hprob = self._hidden_sparsity(x_batch, prev_hprob, hprob, data_len=data_len)
                    pos_hgrad += hg
                    pos_wgrad += wg

                hg, vg, wg = self._get_positive_grad(x_batch, hprob, data_len=data_len)
                pos_hgrad += hg
                pos_vgrad += vg
                pos_wgrad += wg

                # ======== free phase =========
                if self.learning_method == LEARNING_METHOD['cd']:
                    for icd in range(self.num_cd - 1):
                        hsample, vprob, vsample, hprob = self._gibbs_sampling(
                            hprob, sampling=CD_SAMPLING['hidden_visible'], data_len=data_len)
                    hsample, vprob, vsample, hprob = self._gibbs_sampling(
                        hprob, sampling=self.sampling_in_last_cd, data_len=data_len)
                elif self.learning_method == LEARNING_METHOD['pcd']:
                    vprob = self._get_visible_prob(self.pcd_hsample)
                    vsample = self._sample_visible(vprob, data_len=data_len)
                    hprob = self._get_hidden_prob(vsample, data_len=data_len)
                    for ipcd in range(self.num_pcd):
                        self.pcd_hsample, vprob, vsample, hprob = self._gibbs_sampling(
                            hprob, sampling=CD_SAMPLING['hidden_visible'], data_len=data_len)
                    hprob = self.pcd_hsample

                # ======== negative phase =========
                neg_hgrad, neg_vgrad, neg_wgrad = self._get_negative_grad(vsample, hprob,
                                                                          data_len=data_len)

                # update params
                self.hgrad_inc = self.momentum * self.hgrad_inc \
                                 + self.learning_rate_hidden * (pos_hgrad - neg_hgrad)
                self.vgrad_inc = self.momentum * self.vgrad_inc \
                                 + self.learning_rate * (pos_vgrad - neg_vgrad)
                self.wgrad_inc = self.momentum * self.wgrad_inc \
                                 + self.learning_rate * (pos_wgrad - neg_wgrad
                                                         - self.weight_cost * self.w)

                self.h += self.hgrad_inc
                self.v += self.vgrad_inc
                self.w += self.wgrad_inc

                batch_logs.update(self._on_batch_end(x_batch))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _get_hidden_prob(self, vsample, **kwargs):
        data_len = kwargs['data_len'] if 'data_len' in kwargs else np.sum(vsample, axis=1, keepdims=True)
        return sigmoid(vsample.dot(self.w) + np.outer(data_len, self.h))

    def _sample_hidden(self, hprob):
        return (hprob > np.random.rand(*hprob.shape)).astype(np.int)

    def _get_visible_prob(self, hsample):
        # vone = np.exp(hsample.dot(self.w.T) + self.v)
        # return vone / np.sum(vone, axis=1, keepdims=True)
        vone = hsample.dot(self.w.T) + self.v
        return np.exp(vone - logsumexp(vone, axis=1, keepdims=True))

    def _sample_visible(self, vprob, **kwargs):
        data_len = kwargs['data_len'] if 'data_len' in kwargs else np.ones(vprob.shape[0])
        vsample = np.zeros(vprob.shape)
        for i in range(vprob.shape[0]):
            vsample[i] = np.random.multinomial(data_len[i], vprob[i])
        return vsample

    def _hidden_sparsity(self, x, prev_hprob, hprob, **kwargs):
        data_len = kwargs['data_len'] if 'data_len' in kwargs else np.sum(x, axis=1, keepdims=True)
        if prev_hprob.shape == hprob.shape:
            q = self.sparse_decay * prev_hprob + (1 - self.sparse_decay) * hprob
        else:
            q = (1 - self.sparse_decay) * hprob
        prev_hprob = np.copy(hprob)
        sparse_grad = self.sparse_level - q
        hg = self.sparse_weight * np.mean(sparse_grad * data_len, axis=0, keepdims=True)
        # hg = self.sparse_weight * np.mean(sparse_grad, axis=0, keepdims=True)
        wg = self.sparse_weight * x.T.dot(sparse_grad) / x.shape[0]
        return hg, wg, prev_hprob

    def _get_positive_grad(self, x, hprob, **kwargs):
        data_len = kwargs['data_len'] if 'data_len' in kwargs else np.sum(x, axis=1, keepdims=True)
        hg = np.mean(hprob * data_len, axis=0, keepdims=True)
        # hg = np.mean(hprob, axis=0, keepdims=True)
        vg = np.mean(x, axis=0, keepdims=True)
        wg = x.T.dot(hprob) / x.shape[0]
        return hg, vg, wg

    def _get_negative_grad(self, x, hprob, **kwargs):
        return self._get_positive_grad(x, hprob, **kwargs)

    def _gibbs_sampling(self, hprob, sampling=CD_SAMPLING['hidden_visible'], **kwargs):
        if sampling == CD_SAMPLING['hidden']:
            hsample = self._sample_hidden(hprob)
            vprob = self._get_visible_prob(hsample)
            vsample = np.copy(vprob)
            hprob = self._get_hidden_prob(vsample, **kwargs)
        elif sampling == CD_SAMPLING['none']:
            hsample = np.copy(hprob)
            vprob = self._get_visible_prob(hsample)
            data_len = kwargs['data_len'] if 'data_len' in kwargs else np.ones(vprob.shape[0])
            vsample = vprob * data_len
            hprob = self._get_hidden_prob(vsample, **kwargs)
        elif sampling == CD_SAMPLING['hidden_visible']:
            hsample = self._sample_hidden(hprob)
            vprob = self._get_visible_prob(hsample)
            vsample = self._sample_visible(vprob, **kwargs)
            hprob = self._get_hidden_prob(vsample, **kwargs)
        else:
            hsample = np.copy(hprob)
            vprob = self._get_visible_prob(hsample)
            vsample = self._sample_visible(vprob, **kwargs)
            hprob = self._get_hidden_prob(vsample, **kwargs)

        return hsample, vprob, vsample, hprob

    def get_free_energy(self, x, **kwargs):
        data_len = kwargs['data_len'] if 'data_len' in kwargs else np.sum(x, axis=1, keepdims=True)
        wx = x.dot(self.w) + np.outer(data_len, self.h)
        return - x.dot(self.v.T) - logsumone(wx).sum(axis=1, keepdims=True)

    def transform(self, x):
        check_is_fitted(self, "w")
        return self._get_hidden_prob(np.round(np.log(x + 1)))

    def get_params(self, deep=True):
        out = super(ReplicatedSoftmaxRBM, self).get_params(deep=deep)
        param_names = ReplicatedSoftmaxRBM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
