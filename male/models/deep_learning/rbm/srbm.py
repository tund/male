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


class SupervisedRBM(BernoulliBernoulliRBM):
    """ Supervised Restricted Boltzmann Machines
    """

    def __init__(self,
                 model_name="sRBM",
                 inference_engine='variational_inference',
                 *args, **kwargs):
        kwargs["model_name"] = model_name
        super(SupervisedRBM, self).__init__(**kwargs)
        self.inference_engine = inference_engine

    def _init(self):
        super(SupervisedRBM, self)._init()
        try:
            self.inference_engine = INFERENCE_ENGINE[self.inference_engine]
        except KeyError:
            raise ValueError("Inference engine %s is not supported." % self.inference_engine)

    def _init_params(self, x):
        super(SupervisedRBM, self)._init_params(x)

        self.yw_ = self.w_init * np.random.randn(self.num_hidden, self.num_classes_)
        self.yb_ = np.zeros([1, self.num_classes_])
        self.ywgrad_inc_ = np.zeros([self.num_hidden, self.num_classes_])
        self.ybgrad_inc_ = np.zeros([1, self.num_classes_])

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
                y_batch = y[batch_start:batch_end]

                pos_hgrad, pos_vgrad, pos_wgrad = self._initialize_grad()

                # ======= clamp phase ========
                hprob = self._get_hidden_prob(x_batch, y=y_batch)

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
                    pos_ybgrad = np.zeros([1, c])
                    pos_ywgrad = np.zeros([k, c])
                    for i in range(c):
                        idx = (y_batch == i)
                        pos_ywgrad[:, i] += np.sum(hprob[idx].T, axis=1) / batch_size
                        pos_ybgrad[0, i] = np.sum(np.double(idx)) / batch_size
                    neg_ywgrad = hprob.T.dot(softmax(hprob.dot(self.yw_) + self.yb_)) / batch_size
                    neg_ybgrad = np.mean(softmax(hprob.dot(self.yw_) + self.yb_), axis=0, keepdims=True)
                else:
                    pos_ybgrad = np.mean(y_batch, keepdims=True)
                    pos_ywgrad = hprob.T.dot(y_batch) / batch_size
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
                self.hgrad_inc_ = self.momentum_ * self.hgrad_inc_ + self.learning_rate * (pos_hgrad - neg_hgrad)
                self.vgrad_inc_ = self.momentum_ * self.vgrad_inc_ + self.learning_rate * (pos_vgrad - neg_vgrad)
                self.wgrad_inc_ = self.momentum_ * self.wgrad_inc_ \
                                  + self.learning_rate * (pos_wgrad - neg_wgrad - self.weight_cost * self.w_)

                self.ybgrad_inc_ = self.momentum_ * self.ybgrad_inc_ + self.learning_rate * (pos_ybgrad - neg_ybgrad)
                self.ywgrad_inc_ = self.momentum_ * self.ywgrad_inc_ \
                                   + self.learning_rate * (pos_ywgrad - neg_ywgrad - self.weight_cost * self.yw_)

                self.h_ += self.hgrad_inc_
                self.v_ += self.vgrad_inc_
                self.w_ += self.wgrad_inc_

                self.yb_ += self.ybgrad_inc_
                self.yw_ += self.ywgrad_inc_

                batch_logs.update(self._on_batch_end(x_batch, y=y_batch, rdata=vprob))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, y=y_valid)
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

    def _get_hidden_prob(self, x, **kwargs):
        if 'y' in kwargs:
            y = kwargs['y']
            k, c = self.num_hidden, self.num_classes_
            m, n = x.shape[0], self.num_visible

            if self.inference_engine == INFERENCE_ENGINE['variational_inference']:
                ii, jj, diff, num_iters = 0, 1, 1.0, 0

                if self.task == 'classification':
                    constant_term = self.yw_[:, y.astype(np.int8)].T + x.dot(self.w_) + self.h_
                    hprob = [np.random.rand(m, k), np.zeros([m, k])]
                    while diff > 1e-4 and num_iters < 30:
                        e = hprob[ii].dot(self.yw_) + self.yb_
                        me = np.max(e, axis=1, keepdims=True)
                        ee = np.exp(e - me)
                        hprob[jj] = sigmoid(constant_term - ee.dot(self.yw_.T) / ee.sum(axis=1, keepdims=True))
                        diff = np.mean(np.abs(hprob[ii] - hprob[jj]))
                        ii ^= 1
                        jj ^= 1
                        num_iters += 1
                    return hprob[ii]

                else:  # regression
                    # looping at fixed point update
                    constant_term = y.dot(self.yw_.T) + x.dot(self.w_) + self.h_
                    hprob = [sigmoid(constant_term), np.zeros([m, k])]
                    while diff > 1e-4 and num_iters < 100:
                        # first-order Taylor series approximation
                        # hprob[jj] = sigmoid(constant_term - (hprob[ii].dot(self.yw_) + self.yb_).dot(self.yw_.T))
                        # second-order Taylor series approximation
                        hprob[jj] = sigmoid(constant_term - (hprob[ii].dot(self.yw_) + self.yb_).dot(self.yw_.T)
                                            - 0.5 * (self.yw_.T ** 2) * (1 - 2 * hprob[ii]))
                        diff = np.mean(np.abs(hprob[ii] - hprob[jj]))
                        ii ^= 1
                        jj ^= 1
                        num_iters += 1

                    '''
                    # use gradient descent with constraints if the fixed point update is not converged
                    if diff > 1e-4:
                        # LBFGS0 (already checked gradients)
                        optimizer = minimize(self.get_loglikelihood, np.ravel(np.random.rand(M, K)),
                                             bounds=([(0, 1)]*M*K), args=(X, y), jac=self.get_grad, method='L-BFGS-B')
                        return optimizer.x.reshape(M, K)
                    '''

                    return hprob[ii]

            else:  # Gibbs sampling
                num_burnings = 50
                num_samples = 100
                hsample = (0.5 > np.random.rand(m, k)).astype(np.int)
                phk1v = sigmoid(x.dot(self.w_) + self.h_)
                for i in range(num_burnings + num_samples):
                    hprob = np.zeros(m, k)
                    for t in range(k):
                        hsample[:, t] = 0

                        pyhk = np.exp(hsample.dot(self.yw_[:, y.astype(np.int8)]) + self.yb_
                                      - logsumexp(hsample.dot(self.yw_) + self.yb_, axis=1, keepdims=True))
                        hprob0 = pyhk * (1.0 - phk1v[:, t])

                        hsample[:, t] = 1
                        pyhk = np.exp(hsample.dot(self.yw_[:, y.astype(np.int8)]) + self.yb_
                                      - logsumexp(hsample.dot(self.yw_) + self.yb_, axis=1, keepdims=True))
                        hprob1 = pyhk * phk1v[:, t]

                        hprob[:, t] = hprob1 / (hprob0 + hprob1)
                        hsample[:, t] = (hprob[:, t] > np.random.rand(m, 1)).astype(np.int)

                    if i >= num_burnings:
                        raise NotImplementedError
        else:
            return sigmoid(x.dot(self.w_) + self.h_)

    def get_loss(self, x, y, *args, **kwargs):
        if self.task == 'classification':
            y_pred = self.predict_proba(x)
            return log_loss(y, y_pred)
        else:  # regression
            y_pred = self._predict(x)
            return np.sqrt(mean_squared_error(y, y_pred))

    def _get_loglik(self, hprob, x, y):
        if self.task == 'regression':
            hprob = np.clip(hprob.reshape(x.shape[0], self.num_hidden), EPSILON, 1 - EPSILON)
            loglik = np.sum((hprob.dot(self.yw_) + self.yb_).T.dot(y) + np.sum(hprob * (x.dot(self.w_) + self.h_))
                            - 0.5 * np.sum((hprob.dot(self.yw_) + self.yb_) ** 2)
                            - np.sum(hprob * np.log(hprob) + (1 - hprob) * np.log(1 - hprob)))
            return -loglik
        else:
            raise NotImplementedError

    def get_grad(self, hprob, x, y):
        if self.task == 'regression':
            hprob = np.clip(hprob.reshape(x.shape[0], self.num_hidden), EPSILON, 1 - EPSILON)
            df = (y.dot(self.yw_.T) + x.dot(self.w_) + self.h_
                  - (hprob.dot(self.yw_) + self.yb_).dot(self.yw_.T)
                  - np.log(hprob) + np.log(1 - hprob))
            return np.ravel(-df)
        else:
            raise NotImplementedError

    def predict(self, x):
        hpost = self.transform(x)
        if self.task == 'classification':
            return self._decode_labels(np.argmax(softmax(hpost.dot(self.yw_) + self.yb_), axis=1))
        else:
            return hpost.dot(self.yw_) + self.yb_

    def _predict(self, x):
        hpost = self.transform(x)
        if self.task == 'classification':
            return np.argmax(softmax(hpost.dot(self.yw_) + self.yb_), axis=1)
        else:
            return hpost.dot(self.yw_) + self.yb_

    def predict_proba(self, x):
        hpost = self.transform(x)
        if self.task == 'classification':
            return softmax(hpost.dot(self.yw_) + self.yb_)
        else:
            raise NotImplementedError

    def score(self, x, y=None, **kwargs):
        if self.task == 'classification':
            return float(accuracy_score(self.predict(x), y))
        else:
            return -np.sqrt(mean_squared_error(y, self.predict(x)))

    def get_params(self, deep=True):
        out = super(SupervisedRBM, self).get_params(deep=deep)
        out["inference_engine"] = self.inference_engine
        return out

    def get_all_params(self, deep=True):
        out = super(SupervisedRBM, self).get_all_params(deep=deep)
        out.update({
            "yw_": copy.deepcopy(self.yw_),
            "yb_": copy.deepcopy(self.yb_),
            "ywgrad_inc_": copy.deepcopy(self.ywgrad_inc_),
            "ybgrad_inc_": copy.deepcopy(self.ybgrad_inc_)
        })
        return out