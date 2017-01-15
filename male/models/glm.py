"""
Generalized Linear Model
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .. import Model
from ..activations import sigmoid
from ..activations import softmax
from ..utils.generic_utils import make_batches

EPS = np.finfo(np.float32).eps


class GLM(Model):
    """Generalized Linear Model
    """

    def __init__(self,
                 model_name='GLM',
                 link='logit',  # link function
                 loss='logit',  # loss function
                 optimizer='L-BFGS-B',
                 learning_rate=0.01,
                 l2_penalty=0.0,  # ridge regularization
                 l1_penalty=0.0,  # Lasso regularization
                 l1_smooth=1E-5,  # smoothing for Lasso regularization
                 l1_method='pseudo_huber',  # approximation method for L1-norm
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(GLM, self).__init__(**kwargs)
        self.link = link
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l1_penalty = l1_penalty
        self.l1_method = l1_method
        self.l1_smooth = l1_smooth
        self.l2_penalty = l2_penalty

    def _init(self):
        super(GLM, self)._init()
        self.w_ = None
        self.b_ = None
        self.onehot_encoder_ = None

    def _init_params(self, x):
        # initialize weights
        if self.n_classes_ > 2:
            self.w_ = 0.01 * self.random_state_.randn(x.shape[1], self.n_classes_)
            self.b_ = np.zeros(self.n_classes_)
        else:
            self.w_ = 0.01 * self.random_state_.randn(x.shape[1])
            self.b_ = np.zeros(1)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        if self.optimizer == 'L-BFGS-B':
            optimizer = minimize(self._get_loss_check_grad, self._roll_params(), args=(x, y),
                                 jac=self._get_grad_check_grad, method=self.optimizer,
                                 options={'disp': (self.verbose != 0)})
            self.w_, self.b_ = self._unroll_params(optimizer.x)
        if self.optimizer == 'sgd':
            while self.epoch_ < self.num_epochs:
                callbacks.on_epoch_begin(self.epoch_)

                batches = make_batches(x.shape[0], self.batch_size)
                epoch_logs = {}

                for batch_idx, (batch_start, batch_end) in enumerate(batches):
                    batch_logs = {'batch': batch_idx,
                                  'size': batch_end - batch_start + 1}
                    callbacks.on_batch_begin(batch_idx, batch_logs)

                    x_batch = x[batch_start:batch_end]
                    y_batch = y[batch_start:batch_end]

                    dw, db = self.get_grad(x_batch, y_batch)

                    self.w_ -= self.learning_rate * dw
                    self.b_ -= self.learning_rate * db

                    outs = self._on_batch_end(x_batch, y_batch)
                    for l, o in zip(self.metrics, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_idx, batch_logs)

                if do_validation:
                    outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                    for l, o in zip(self.metrics, outs):
                        epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(self.epoch_, epoch_logs)

                self.epoch_ += 1
                if self.stop_training_:
                    self.epoch_ = self.stop_training_
                    break

    def _encode_labels(self, y):
        y = super(GLM, self)._encode_labels(y)
        if self.loss == 'softmax':
            self.onehot_encoder_ = OneHotEncoder()
            y = self.onehot_encoder_.fit_transform(y.reshape(-1, 1)).toarray()
        return y

    def _decode_labels(self, y):
        if self.loss == 'softmax':
            y = np.argmax(y, axis=1)
        return super(GLM, self)._decode_labels(y)

    def _transform_labels(self, y):
        y = super(GLM, self)._transform_labels(y)
        if self.loss == 'softmax':
            y = self.onehot_encoder_.transform(y.reshape(-1, 1)).toarray()
        return y

    def get_loss(self, x, y, *args, **kwargs):
        w = kwargs['w'] if 'w' in kwargs else self.w_
        b = kwargs['b'] if 'b' in kwargs else self.b_

        # compute p(y|x) basing on link function
        t = self.get_link(x, w=w, b=b)

        # compute loss / (negative) log-likelihood
        f = 0
        if self.loss == 'logit':
            t = np.maximum(t, EPS)
            t = np.minimum(t, 1 - EPS)
            f = - np.sum(y.T.dot(np.log(t)) + (1 - y).T.dot(np.log(1 - t)))
        elif self.loss == 'softmax':
            t = np.maximum(t, EPS)
            f = - np.sum(y * np.log(t))
        elif self.loss == 'quadratic':
            f = 0.5 * np.sum((y - t) ** 2)
        f /= x.shape[0]

        if self.l2_penalty > 0:
            f += 0.5 * self.l2_penalty * np.sum(w ** 2)

        if self.l1_penalty > 0:
            if self.l1_method == 'pseudo_huber':
                w2sqrt = np.sqrt(w ** 2 + self.l1_smooth ** 2)
                f += self.l1_penalty * np.sum(w2sqrt)
        return f

    def get_link(self, x, *args, **kwargs):
        w = kwargs['w'] if 'w' in kwargs else self.w_
        b = kwargs['b'] if 'b' in kwargs else self.b_

        y = x.dot(w) + b
        if self.link == 'logit':
            y = sigmoid(y)
        elif self.link == 'softmax':
            y = softmax(y)
        return y

    def get_grad(self, x, y, *args, **kwargs):
        w = kwargs['w'] if 'w' in kwargs else self.w_
        b = kwargs['b'] if 'b' in kwargs else self.b_

        # compute p(y|x) basing on link function
        t = self.get_link(x, w=w, b=b)

        dw = - x.T.dot(y - t) / x.shape[0]
        db = - np.mean(y - t, axis=0)

        if self.l2_penalty > 0:
            dw += self.l2_penalty * w

        if self.l1_penalty > 0:
            if self.l1_method == 'pseudo_huber':
                w2sqrt = np.sqrt(w ** 2 + self.l1_smooth ** 2)
                dw += self.l1_penalty * w / w2sqrt

        return dw, db

    def _get_loss_check_grad(self, w, x, y):
        w, b = self._unroll_params(w)
        return self.get_loss(x, y, w=w, b=b)

    def _get_grad_check_grad(self, w, x, y):
        w, b = self._unroll_params(w)
        dw, db = self.get_grad(x, y, w=w, b=b)
        return np.concatenate([np.ravel(dw), np.ravel(db)])

    def predict(self, x):
        check_is_fitted(self, "w_")
        check_is_fitted(self, "b_")

        y = self.predict_proba(x)
        if self.loss == 'logit':
            y = np.require(y >= 0.5, dtype=np.uint8)
        # elif self.loss == 'softmax':
        #     y = np.argmax(y, axis=1)
        return self._decode_labels(y)

    def predict_proba(self, x):
        check_is_fitted(self, "w_")
        check_is_fitted(self, "b_")

        return self.get_link(x)

    def _roll_params(self):
        return np.concatenate([super(GLM, self)._roll_params(),
                               np.ravel(self.w_),
                               np.ravel(self.b_)])

    def _unroll_params(self, w):
        ww = super(GLM, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww])
        w_ = w[idx:idx + self.w_.size].reshape(self.w_.shape)
        idx += self.w_.size
        b_ = w[idx:idx + self.b_.size].reshape(self.b_.shape)
        return ww + (w_, b_)

    def get_params(self, deep=True):
        out = super(GLM, self).get_params(deep=deep)
        out.update({
            'link': self.link,
            'loss': self.loss,
            'optimizer': self.optimizer,
            'l1_penalty': self.l1_penalty,
            'l1_method': self.l1_method,
            'l1_smooth': self.l1_smooth,
            'l2_penalty': self.l2_penalty,
            'learning_rate': self.learning_rate,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(GLM, self).get_all_params(deep=deep)
        out.update({
            'w_': copy.deepcopy(self.w_),
            'b_': copy.deepcopy(self.b_),
            'onehot_encoder_': copy.deepcopy(self.onehot_encoder_),
        })
        return out
