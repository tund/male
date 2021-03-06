from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from scipy.misc import logsumexp
from sklearn.preprocessing import StandardScaler

from .bbrbm import BernoulliBernoulliRBM
from .rbm import INFERENCE_ENGINE, LEARNING_METHOD, CD_SAMPLING
from ....utils.generic_utils import make_batches
from ....utils.func_utils import sigmoid, softmax

EPSILON = np.finfo(np.float32).eps
APPROX_METHOD = {'first_order': 0, 'second_order': 1}


class SupervisedRBM(BernoulliBernoulliRBM):
    """ Supervised Restricted Boltzmann Machines
    """

    def __init__(self,
                 model_name="sRBM",
                 inference_engine='variational_inference',
                 approx_method='first_order',  # {'first_order', 'second_order'}
                 **kwargs):
        super(SupervisedRBM, self).__init__(model_name=model_name, **kwargs)
        self.inference_engine = inference_engine
        self.approx_method = approx_method

    def _init(self):
        super(SupervisedRBM, self)._init()
        try:
            self.inference_engine = INFERENCE_ENGINE[self.inference_engine]
        except KeyError:
            raise ValueError("Inference engine %s is not supported." % self.inference_engine)
        try:
            self.approx_method = APPROX_METHOD[self.approx_method]
        except KeyError:
            raise ValueError("Approximation method %s is not supported." % self.approx_method)

    def _init_params(self, x):
        super(SupervisedRBM, self)._init_params(x)

        c = self.num_classes if self.task == 'classification' else 1
        self.yw = self.w_init * np.random.randn(self.num_hidden, c)  # [K,C]
        self.yb = np.zeros([1, c])  # [1,C]
        self.ywgrad_inc = np.zeros([self.num_hidden, c])  # [K,C]
        self.ybgrad_inc = np.zeros([1, c])  # [1,C]
        # variational parameters for posterior p(h|v,y) inference, where
        # mu[i] = p(h[i]=1|v,y)
        self.mu = np.random.rand(self.batch_size, self.num_hidden)  # [BS,K]

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        k, n, c = self.num_hidden, self.num_visible, self.num_classes
        prev_hprob = np.zeros([1, k])

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

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
                    neg_ywgrad = hprob.T.dot(softmax(hprob.dot(self.yw) + self.yb)) / batch_size
                    neg_ybgrad = np.mean(softmax(hprob.dot(self.yw) + self.yb),
                                         axis=0, keepdims=True)
                else:
                    pos_ybgrad = np.mean(y_batch, keepdims=True)
                    pos_ywgrad = hprob.T.dot(y_batch[:, np.newaxis]) / batch_size
                    neg_ybgrad = np.mean(hprob.dot(self.yw) + self.yb, keepdims=True)
                    neg_ywgrad = hprob.T.dot(hprob.dot(self.yw) + self.yb) / batch_size

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
                self.hgrad_inc = self.momentum * self.hgrad_inc \
                                 + self.learning_rate * (pos_hgrad - neg_hgrad)
                self.vgrad_inc = self.momentum * self.vgrad_inc \
                                 + self.learning_rate * (pos_vgrad - neg_vgrad)
                self.wgrad_inc = self.momentum * self.wgrad_inc \
                                 + self.learning_rate * (pos_wgrad - neg_wgrad
                                                         - self.weight_cost * self.w)

                self.ybgrad_inc = self.momentum * self.ybgrad_inc \
                                  + self.learning_rate * (pos_ybgrad - neg_ybgrad)
                self.ywgrad_inc = self.momentum * self.ywgrad_inc \
                                  + self.learning_rate * (pos_ywgrad - neg_ywgrad
                                                          - self.weight_cost * self.yw)

                self.h += self.hgrad_inc
                self.v += self.vgrad_inc
                self.w += self.wgrad_inc

                self.yb += self.ybgrad_inc
                self.yw += self.ywgrad_inc

                batch_logs.update(self._on_batch_end(x_batch, y=y_batch, rdata=vprob))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, y=self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _get_hidden_prob(self, x, **kwargs):
        if 'y' in kwargs:
            y = kwargs['y']
            k, c = self.num_hidden, self.num_classes
            m, n = x.shape[0], self.num_visible

            if self.inference_engine == INFERENCE_ENGINE['variational_inference']:
                ii, jj, diff, num_iters = 0, 1, 1.0, 0

                if self.task == 'classification':
                    # constant_term: [M,K]
                    constant_term = self.yw[:, y.astype(np.int8)].T + x.dot(self.w) + self.h
                    yw_yw = self.yw.T * self.yw.T  # [C,K]
                    hprob = [np.random.rand(m, k), np.zeros([m, k])]  # [M, K]
                    # looping at fixed point update
                    while diff > 1e-4 and num_iters < 30:
                        e = hprob[ii].dot(self.yw) + self.yb  # [M,C]
                        me = np.max(e, axis=1, keepdims=True)  # [M,1]
                        ee = np.exp(e - me)  # [M,C]
                        # first-order Taylor series approximation
                        if self.approx_method == APPROX_METHOD['first_order']:
                            hprob[jj] = sigmoid(constant_term - ee.dot(self.yw.T)
                                                / ee.sum(axis=1, keepdims=True))
                        # second-order Taylor series approximation
                        else:
                            lbd = ee / ee.sum(axis=1, keepdims=True)  # [M,C]
                            db_mu = ee.dot(self.yw.T) / ee.sum(axis=1, keepdims=True)  # [M,K]
                            dlbd_mu = lbd.sum(axis=0, keepdims=True) * self.yw \
                                      - db_mu.T.dot(lbd)  # [K,C]
                            hh = hprob[ii] * (1 - hprob[ii])  # [M,K]
                            term1 = 0.5 * hh.dot(dlbd_mu.dot(yw_yw))
                            term2 = 0.5 * (1 - 2 * hprob[ii]) * lbd.dot(yw_yw)  # [M,K]
                            term3 = (hh * lbd.dot(self.yw.T)).dot(self.yw.dot(dlbd_mu.T))
                            term4 = 0.5 * (1 - 2 * hprob[ii]) * lbd.dot(self.yw.T)
                            hprob[jj] = sigmoid(constant_term - db_mu
                                                - term1 - term2 + term3 + term4)
                        diff = np.mean(np.abs(hprob[ii] - hprob[jj]))
                        ii ^= 1
                        jj ^= 1
                        num_iters += 1
                    return hprob[ii]

                else:  # regression
                    # constant_term: [M,K]
                    constant_term = np.outer(y, self.yw) + x.dot(self.w) + self.h
                    hprob = [sigmoid(constant_term), np.zeros([m, k])]
                    # looping at fixed point update
                    while diff > 1e-4 and num_iters < 100:
                        # first-order Taylor series approximation
                        if self.approx_method == APPROX_METHOD['first_order']:
                            hprob[jj] = sigmoid(constant_term - (hprob[ii].dot(self.yw)
                                                                 + self.yb).dot(self.yw.T))

                        # second-order Taylor series approximation
                        else:
                            hprob[jj] = sigmoid(constant_term - (hprob[ii].dot(self.yw)
                                                                 + self.yb).dot(self.yw.T)
                                                - 0.5 * (self.yw.T ** 2) * (1 - 2 * hprob[ii]))
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
                num_samples = 1
                hsample = (0.5 > np.random.rand(m, k)).astype(np.int)  # [M,K]
                # p(h_k = 1 | v)
                phk1v = sigmoid(x.dot(self.w) + self.h)  # [M,K]

                if self.task == 'classification':
                    for i in range(num_burnings + num_samples):
                        hprob = np.zeros([m, k])  # [M,K]
                        for t in range(k):
                            hsample[:, t] = 0
                            # p(y | h_k = 0, h_(-k)): [M,1]
                            pyhk = np.exp(np.sum(hsample * self.yw[:, y.astype(np.int8)].T,
                                                 axis=1, keepdims=True)
                                          + self.yb[:, y.astype(np.int8)].T
                                          - logsumexp(hsample.dot(self.yw) + self.yb,
                                                      axis=1, keepdims=True))
                            # p(h_k = 0 | h_(-k), v, y) = p(y | h_k = 0, h_(-k)) * p(h_k = 0 | v)
                            # Note that p(h_k = 0 | v) = 1 - p(h_k = 1 | v)
                            hprob0 = pyhk * (1.0 - phk1v[:, [t]])  # [M,1]

                            hsample[:, t] = 1
                            # p(y | h_k = 1, h_(-k)): [M,1]
                            pyhk = np.exp(np.sum(hsample * self.yw[:, y.astype(np.int8)].T,
                                                 axis=1, keepdims=True)
                                          + self.yb[:, y.astype(np.int8)].T
                                          - logsumexp(hsample.dot(self.yw) + self.yb,
                                                      axis=1, keepdims=True))
                            # p(h_k = 1 | h_(-k), v, y) = p(y | h_k = 1, h_(-k)) * p(h_k = 1 | v)
                            hprob1 = pyhk * phk1v[:, [t]]  # [M,1]

                            hprob[:, [t]] = hprob1 / (hprob0 + hprob1)  # [M,1]
                            hsample[:, t] = (hprob[:, t] > np.random.rand(m)).astype(
                                np.int)  # [M,1]
                else:  # regression
                    for i in range(num_burnings + num_samples):
                        hprob = np.zeros([m, k])  # [M,K]
                        for t in range(k):
                            hsample[:, t] = 0
                            # p(y | h_k = 0, h_(-k)): [M,1]
                            pyhk = np.exp(-0.5 * (y[:, np.newaxis] ** 2)
                                          + (hsample.dot(self.yw) + self.yb) * y[:, np.newaxis]
                                          - 0.5 * (hsample.dot(self.yw) + self.yb) ** 2)
                            # p(h_k = 0 | h_(-k), v, y) = p(y | h_k = 0, h_(-k)) * p(h_k = 0 | v)
                            # Note that p(h_k = 0 | v) = 1 - p(h_k = 1 | v)
                            hprob0 = pyhk * (1.0 - phk1v[:, [t]])  # [M,1]

                            hsample[:, t] = 1
                            # p(y | h_k = 1, h_(-k)): [M,1]
                            pyhk = np.exp(-0.5 * (y[:, np.newaxis] ** 2)
                                          + (hsample.dot(self.yw) + self.yb) * y[:, np.newaxis]
                                          - 0.5 * (hsample.dot(self.yw) + self.yb) ** 2)
                            # p(h_k = 1 | h_(-k), v, y) = p(y | h_k = 1, h_(-k)) * p(h_k = 1 | v)
                            hprob1 = pyhk * phk1v[:, [t]]  # [M,1]

                            hprob[:, [t]] = hprob1 / (hprob0 + hprob1)  # [M,1]
                            hsample[:, t] = (hprob[:, t] > np.random.rand(m)).astype(
                                np.int)  # [M,1]

                return hprob  # endif
        else:
            return sigmoid(x.dot(self.w) + self.h)

    def _predict_from_hidden(self, hidden):
        if self.task == 'classification':
            return np.argmax(softmax(hidden.dot(self.yw) + self.yb), axis=1)
        else:
            return hidden.dot(self.yw) + self.yb

    def _predict_proba_from_hidden(self, hidden):
        if self.task == 'classification':
            return softmax(hidden.dot(self.yw) + self.yb)
        else:
            return hidden.dot(self.yw) + self.yb

    def get_loss(self, x, y, *args, **kwargs):
        if self.task == 'classification':
            y_pred = self.predict_proba(x)
            return log_loss(y, y_pred, labels=self.label_encoder.classes_)
        else:  # regression
            y_pred = self._predict(x)
            return np.sqrt(mean_squared_error(y, y_pred))

    def _get_loglik(self, hprob, x, y):
        if self.task == 'regression':
            hprob = np.clip(hprob.reshape(x.shape[0], self.num_hidden), EPSILON, 1 - EPSILON)
            loglik = np.sum((hprob.dot(self.yw) + self.yb).T.dot(y)
                            + np.sum(hprob * (x.dot(self.w) + self.h))
                            - 0.5 * np.sum((hprob.dot(self.yw) + self.yb) ** 2)
                            - np.sum(hprob * np.log(hprob) + (1 - hprob) * np.log(1 - hprob)))
            return -loglik
        else:
            raise NotImplementedError

    def get_grad(self, hprob, x, y):
        if self.task == 'regression':
            hprob = np.clip(hprob.reshape(x.shape[0], self.num_hidden), EPSILON, 1 - EPSILON)
            df = (y.dot(self.yw.T) + x.dot(self.w) + self.h
                  - (hprob.dot(self.yw) + self.yb).dot(self.yw.T)
                  - np.log(hprob) + np.log(1 - hprob))
            return np.ravel(-df)
        else:
            raise NotImplementedError

    def _get_loss_check_grad(self, w, x, y):
        pass
        h = self._unroll_params(w)
        if self.task == 'regression':
            if self.approx_method == APPROX_METHOD['first_order']:
                pass
            else:
                pass

    def _get_grad_check_grad(self, w, x, y):
        pass
        hprob = self._unroll_params(w)
        if self.task == 'classification':
            shared_term = self.yw[:, y.astype(np.int8)].T + x.dot(self.w) + self.h
            if self.approx_method == APPROX_METHOD['first_order']:
                e = hprob.dot(self.yw) + self.yb
                me = np.max(e, axis=1, keepdims=True)
                ee = np.exp(e - me)
                dw = shared_term - ee.dot(self.yw.T) / ee.sum(axis=1, keepdims=True)
            else:  # second-order
                raise NotImplementedError
        if self.task == 'regression':
            shared_term = y.dot(self.yw.T) + x.dot(self.w) + self.h
            if self.approx_method == APPROX_METHOD['first_order']:
                dw = shared_term - (hprob.dot(self.yw) + self.yb).dot(self.yw.T)
            else:  # second-order
                dw = shared_term - (hprob.dot(self.yw) + self.yb).dot(self.yw.T) \
                     - 0.5 * (self.yw.T ** 2) * (1 - 2 * hprob)
        return np.ravel(dw)

    def _unroll_params(self, w):
        return w.reshape([self.batch_size, self.num_hidden])

    def _roll_params(self):
        return np.ravel(self.mu)

    def _encode_labels(self, y):
        yy = super(SupervisedRBM, self)._encode_labels(y)
        if self.task == 'regression':
            self.label_encoder_ = StandardScaler()
            yy = self.label_encoder_.fit_transform(yy)
        return yy

    def _decode_labels(self, y):
        y = super(SupervisedRBM, self)._decode_labels(y)
        if self.task == 'regression':
            return self.label_encoder_.inverse_transform(y)
        else:
            return y

    def _transform_labels(self, y):
        y = super(SupervisedRBM, self)._transform_labels(y)
        if self.task == 'regression':
            return self.label_encoder_.transform(y)
        else:
            return y

    def predict(self, x):
        return self._decode_labels(self._predict(x))

    def _predict(self, x):
        hpost = self.transform(x)
        if self.task == 'classification':
            return np.argmax(softmax(hpost.dot(self.yw) + self.yb), axis=1)
        else:
            return hpost.dot(self.yw) + self.yb

    def predict_proba(self, x):
        hpost = self.transform(x)
        if self.task == 'classification':
            return softmax(hpost.dot(self.yw) + self.yb)
        else:
            raise NotImplementedError

    def score(self, x, y=None, **kwargs):
        if self.exception:
            return - (1e+8)
        elif self.task == 'classification':
            return float(accuracy_score(self.predict(x), y))
        else:
            return -mean_squared_error(y, self.predict(x))

    def get_params(self, deep=True):
        out = super(SupervisedRBM, self).get_params(deep=deep)
        param_names = SupervisedRBM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
