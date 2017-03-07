from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

from ... import Model


class KSGD(Model):
    """Kernel Stochastic Gradient Descent
    """

    def __init__(self,
                 model_name="KernelSGD",
                 lbd=1.0,
                 eps=0.1,
                 gamma=0.1,
                 kernel='gaussian',
                 loss='hinge',
                 batch_size=100,
                 avg_weight=False,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(KSGD, self).__init__(**kwargs)
        self.loss = loss
        self.eps = eps
        self.kernel = kernel
        self.gamma = gamma  # kernel width
        self.lbd = lbd  # regularization parameter
        self.batch_size = batch_size
        self.avg_weight = avg_weight

    def _init(self):
        super(KSGD, self)._init()
        self.w_ = None
        self.x_ = None
        if self.loss == 'hinge' or self.loss == 'logit':
            self.task = 'classification'
        else:
            self.task = 'regression'

    def _init_params(self, x):
        if self.num_classes_ > 2:
            self.w_ = 0.01 * self.random_state_.randn(x.shape[0], self.num_classes_)
        else:
            self.w_ = 0.01 * self.random_state_.randn(x.shape[0])

    def _get_wx(self, t, x):
        if t == 0:
            return [0]
        else:
            if self.kernel == 'gaussian':
                xx = (self.x_[:t, :] - x)
                if self.num_classes_ > 2:
                    return np.sum(
                        self.w_[:t] * np.exp(-self.gamma * np.sum(xx * xx, axis=1, keepdims=True)),
                        axis=0)
                else:
                    return np.sum(
                        self.w_[:t, np.newaxis] * np.exp(-self.gamma * np.sum(xx * xx, axis=1, keepdims=True)),
                        axis=0)
            else:
                return [0]

    def _get_wxy(self, t, x, y):
        if t == 0:
            return (0, -1)
        else:
            if self.kernel == 'gaussian':
                xx = (self.x_[:t, :] - x)
                k = np.sum(
                    self.w_[:t, :] * np.exp(-self.gamma * np.sum(xx * xx, axis=1, keepdims=True)),
                    axis=0)
                idx = np.ones(self.num_classes_, np.bool)
                idx[y] = False
                z = np.argmax(k[idx])
                z += (z >= y)
                return (k[y] - k[z], z)
            else:
                return (0, -1)

    def get_grad(self, t, x, y):
        if self.num_classes_ > 2:
            wxy, z = self._get_wxy(t, x, y)
            if self.loss == 'hinge':
                return (-1, z) if wxy <= 1 else (0, z)
            else:
                return (-1 / (1 + np.exp(wxy)), z)
        else:
            wx = self._get_wx(t, x)[0]
            if self.loss == 'hinge':
                return (-y, -1) if y * wx <= 1 else (0, -1)
            elif self.loss == 'l1':
                return (np.sign(wx - y), -1)
            elif self.loss == 'l2':
                return (wx - y, -1)
            elif self.loss == 'logit':
                if y * wx > 0:
                    return (-y * np.exp(-y * wx) / (np.exp(-y * wx) + 1), -1)
                else:
                    return (-y / (1 + np.exp(y * wx)), -1)
            elif self.loss == 'eps_intensive':
                return (np.sign(wx - y), -1) if np.abs(y - wx) > self.eps else (0, -1)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        n = x.shape[0]

        if self.avg_weight:
            w_avg = np.zeros(self.w_.shape)

        self.x_ = x
        for t in range(n):
            alpha_t, z = self.get_grad(t, x[t], y[t])  # compute \alpha_t
            self.w_ *= (1.0 * t) / (t + 1)
            if self.num_classes_ > 2:
                self.w_[t, y[t]] = -alpha_t / (self.lbd * (t + 1))
                if z >= 0:
                    self.w_[t, z] = alpha_t / (self.lbd * (t + 1))
            else:
                self.w_[t] = -alpha_t / (self.lbd * (t + 1))

            if self.avg_weight:
                w_avg += self.w_

        if self.avg_weight:
            self.w_ = w_avg / n

    def _encode_labels(self, y):
        yy = y.copy()
        yy = super(KSGD, self)._encode_labels(yy)
        if self.num_classes_ == 2:
            yy[yy == 0] = -1
        return yy

    def _decode_labels(self, y):
        yy = y.copy()
        if self.num_classes_ == 2:
            yy[yy == -1] = 0
        return super(KSGD, self)._decode_labels(yy)

    def _transform_labels(self, y):
        yy = y.copy()
        yy = super(KSGD, self)._transform_labels(yy)
        if self.num_classes_ == 2:
            yy[yy == 0] = -1
        return yy

    def predict(self, x):
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            wx = self._get_wx(len(self.w_), x[i])
            if self.task == 'classification':
                if self.num_classes_ == 2:
                    y[i] = self._decode_labels(np.uint8(wx >= 0))
                else:
                    y[i] = self._decode_labels(np.argmax(wx))
            else:
                y[i] = wx[0]
        return y

    def get_params(self, deep=True):
        out = super(KSGD, self).get_params(deep=deep)
        out.update({
            'loss': self.loss,
            'eps': self.eps,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'lbd': self.lbd,
            'batch_size': self.batch_size,
            'avg_weight': self.avg_weight
        })
        return out

    def get_all_params(self, deep=True):
        out = super(KSGD, self).get_all_params(deep=deep)
        out.update({
            'w_': copy.deepcopy(self.w_),
            'x_': copy.deepcopy(self.x_),
        })
        return out
