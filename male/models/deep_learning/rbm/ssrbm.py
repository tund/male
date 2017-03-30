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

from .srbm import SupervisedRBM
from .rbm import INFERENCE_ENGINE, LEARNING_METHOD, CD_SAMPLING
from ....utils.generic_utils import make_batches
from ....utils.func_utils import sigmoid, softmax

EPSILON = np.finfo(np.float32).eps
APPROX_METHOD = {'first_order': 0, 'second_order': 1}


class SemiSupervisedRBM(SupervisedRBM):
    """ Semi-Supervised Restricted Boltzmann Machines
    """

    def __init__(self,
                 model_name="ssRBM",
                 *args, **kwargs):
        kwargs["model_name"] = model_name
        super(SemiSupervisedRBM, self).__init__(**kwargs)

    def _init_params(self, x):
        super(SemiSupervisedRBM, self)._init_params(x)

        c = self.num_classes_ if self.task == 'classification' else 1
        self.yw_ = self.w_init * np.random.randn(self.num_hidden, c)  # [K,C]
        self.yb_ = np.zeros([1, c])  # [1,C]
        self.ywgrad_inc_ = np.zeros([self.num_hidden, c])  # [K,C]
        self.ybgrad_inc_ = np.zeros([1, c])  # [1,C]
        # variational parameters for posterior p(h|v,y) inference, where
        # mu[i] = p(h[i]=1|v,y)
        self.mu_ = np.random.rand(self.batch_size, self.num_hidden)  # [BS,K]

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        k, n, c = self.num_hidden, self.num_visible, self.num_classes_
        prev_hprob = np.zeros([1, k])

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch_ < self.num_epochs) and (not self.stop_training_):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch_)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                batch_size = batch_end - batch_start
                x_batch = x[batch_start:batch_end]
                y_batch = np.copy(y[batch_start:batch_end])
                idx0 = np.where(y_batch == 10 ** 8)[0]  # missing labels
                idx1 = np.where(y_batch != 10 ** 8)[0]  # containing labels

                pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

                # ======= clamp phase ========
                hprob = np.zeros([batch_size, k])
                hprob[idx1] = self._get_hidden_prob(x_batch[idx1], y=y_batch[idx1])
                hprob[idx0] = self._get_hidden_prob(x_batch[idx0])
                y_pred = self._predict_proba_from_hidden(hprob[idx0])
                y_batch[idx0] = self._predict_from_hidden(hprob[idx0])

                # sparsity
                if self.sparse_weight > 0:
                    hg, wg, prev_hprob = self._hidden_sparsity(x_batch, prev_hprob, hprob)
                    pos_hgrad += hg
                    pos_wgrad += wg

                hg, vg, wg = self._get_positive_grad(x_batch, hprob)
                pos_hgrad += hg
                pos_vgrad += vg
                pos_wgrad += wg

                # gradients for the label
                if self.task == 'classification':
                    # gradients from imputed labels
                    pos_ybgrad = np.sum(y_pred, axis=0, keepdims=True)
                    pos_ywgrad = hprob[idx0].T.dot(y_pred)
                    # gradients from true labels
                    for i in range(c):
                        idx = (y_batch[idx1] == i)
                        pos_ywgrad[:, i] += np.sum(hprob[idx1][idx].T, axis=1)
                        pos_ybgrad[0, i] += np.sum(np.double(idx))
                    pos_ybgrad /= batch_size
                    pos_ywgrad /= batch_size
                    neg_ywgrad = hprob.T.dot(softmax(hprob.dot(self.yw_) + self.yb_)) / batch_size
                    neg_ybgrad = np.mean(softmax(hprob.dot(self.yw_) + self.yb_),
                                         axis=0, keepdims=True)
                else:
                    pos_ybgrad = np.mean(y_batch, keepdims=True)
                    pos_ywgrad = hprob.T.dot(y_batch[:, np.newaxis]) / batch_size
                    neg_ybgrad = np.mean(hprob.dot(self.yw_) + self.yb_, keepdims=True)
                    neg_ywgrad = hprob.T.dot(hprob.dot(self.yw_) + self.yb_) / batch_size

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

                self.ybgrad_inc_ = self.momentum_ * self.ybgrad_inc_ \
                                   + self.learning_rate * (pos_ybgrad - neg_ybgrad)
                self.ywgrad_inc_ = self.momentum_ * self.ywgrad_inc_ \
                                   + self.learning_rate * (pos_ywgrad - neg_ywgrad
                                                           - self.weight_cost * self.yw_)

                self.h_ += self.hgrad_inc_
                self.v_ += self.vgrad_inc_
                self.w_ += self.wgrad_inc_

                self.yb_ += self.ybgrad_inc_
                self.yw_ += self.ywgrad_inc_

                batch_logs.update(self._on_batch_end(x_batch, y=y_batch, rdata=vprob))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, y=self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

    def get_loss(self, x, y, *args, **kwargs):
        if self.task == 'classification':
            y_pred = self.predict_proba(x)
            y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
            return np.mean(-np.log(y_pred[range(len(y)), y]))
        else:  # regression
            y_pred = self._predict(x)
            return np.sqrt(mean_squared_error(y, y_pred))

    def _encode_labels(self, y):
        idx1 = np.where(y != 10 ** 8)[0]  # containing labels
        yy = super(SupervisedRBM, self)._encode_labels(y[idx1])
        if self.task == 'regression':
            self.label_encoder_ = StandardScaler()
            yy = self.label_encoder_.fit_transform(yy)
        y[idx1] = yy
        if self.task == 'classification':
            y = y.astype(np.int32)
        return y

    def _decode_labels(self, y):
        idx1 = np.where(y != 10 ** 8)[0]  # containing labels
        yy = super(SupervisedRBM, self)._decode_labels(y[idx1])
        if self.task == 'regression':
            yy = self.label_encoder_.inverse_transform(yy)
        y[idx1] = yy
        return y

    def _transform_labels(self, y):
        idx1 = np.where(y != 10 ** 8)[0]  # containing labels
        yy = super(SupervisedRBM, self)._transform_labels(y[idx1])
        if self.task == 'regression':
            yy = self.label_encoder_.transform(yy)
        y[idx1] = yy
        if self.task == 'classification':
            y = y.astype(np.int32)
        return y
