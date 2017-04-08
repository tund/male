from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np
from numpy.linalg import norm

from .sgd import KSGD

INF = 1e+8
MAINTENANCE = {'merging': 0, 'removal': 1, 'k-merging': 2}


class DualSGD(KSGD):
    """Dual Space Gradient Descent
    """

    def __init__(self,
                 model_name="DualSGD",
                 k=10,  # number of support vectors to be merged
                 D=100,  # number of random features
                 max_budget_size=1000,
                 maintain='k-merging',
                 record=-1,
                 **kwargs):
        super(DualSGD, self).__init__(model_name=model_name, **kwargs)
        self.k = k
        self.D = D
        self.max_budget_size = max_budget_size
        self.maintain = maintain
        self.record = record

    def _init(self):
        super(DualSGD, self)._init()

        # verify the maintenance strategy
        if not self.maintain in MAINTENANCE:
            raise ValueError("Maintenance strategy %s is not supported." % self.maintain)

        self.budget_size = 0
        self.rw = None  # weights for random features
        self.idx = None

        self.list_train_time = []
        self.list_mistake = []
        self.list_mistake_idx = []
        self.mistake = 0.0

    def _init_params(self, x):
        if self.num_classes > 2:
            self.w = np.zeros([self.max_budget_size, self.num_classes])
            self.rw = np.zeros([2 * self.D, self.num_classes])
        else:
            self.w = np.zeros([self.max_budget_size, 1])
            self.rw = np.zeros([2 * self.D, 1])
        self.idx = np.zeros(self.max_budget_size, dtype=np.int)

    def get_wx(self, x, x0, rx):
        b = self.budget_size
        if b == 0:
            return np.array([0])
        else:
            if self.kernel == 'gaussian':
                xx = (x[self.idx[:b]] - x0)
                return np.sum(
                    self.w[:b] * np.exp(-self.gamma * (xx * xx).sum(axis=1, keepdims=True)),
                    axis=0) + rx.dot(self.rw)
            else:
                return np.array([0])

    def get_wxy(self, x, x0, rx, y, wx=None):
        if self.budget_size == 0:
            return (0, -1)
        else:
            if self.kernel == 'gaussian':
                if wx is None:
                    wx = self.get_wx(x, x0, rx)
                idx = np.ones(self.num_classes, np.bool)
                idx[y] = False
                z = np.argmax(wx[idx])
                z += (z >= y)
                return (wx[y] - wx[z], z)
            else:
                return (0, -1)

    def get_grad(self, x, x0, rx, y, wx=None):
        if self.num_classes > 2:
            wxy, z = self.get_wxy(x, x0, rx, y, wx)
            if self.loss == 'hinge':
                return (-1, z) if wxy <= 1 else (0, z)
            else:  # logit loss
                if wxy > 0:
                    return (-np.exp(-wxy) / (np.exp(-wxy) + 1), z)
                else:
                    return (-1 / (1 + np.exp(wxy)), z)
        else:
            wx = self.get_wx(x, x0, rx)[0] if wx is None else wx[0]
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
            elif self.loss == 'eps_insensitive':
                return (np.sign(wx - y), -1) if np.abs(y - wx) > self.eps else (0, -1)

    def add_to_core_set(self, t, w, y, z):
        b = self.budget_size
        self.idx[b] = t
        if self.num_classes > 2:
            self.w[b, y] = w
            if z >= 0:
                self.w[b, z] = -w
        else:
            self.w[b] = w
        self.budget_size += 1

    def remove(self, idx):
        n = len(idx)
        mask = np.ones(self.max_budget_size, np.bool)
        mask[idx] = False
        self.w[:-n] = self.w[mask]
        self.idx[:-n] = self.idx[mask]
        self.budget_size -= n
        # self.w_ = np.roll(self.w_, self.size_ - idx - 1)
        # self.idx_ = np.roll(self.idx_, self.size_ - idx - 1)
        # self.budget_size_ -= 1

    def maintain_budget(self, rx, w):
        if self.maintain == 'k-merging':
            i = np.argsort(norm(self.w[:self.budget_size], axis=1))
            # i = np.argpartition(norm(self.w_[:self.size_], axis=1), self.k)
            self.rw += rx[self.idx[i[:self.k]]].T.dot(self.w[i[:self.k]])
            self.remove(i[:self.k])

        elif self.maintain == 'removal':
            i = np.argmin(norm(self.w, axis=1))
            mask = np.ones(self.max_budget_size, np.bool)
            mask[i] = False
            self.sv[:-1] = self.sv[mask]
            self.w[:-1] = self.w[mask]
            self.budget_size -= 1

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        y0 = self._decode_labels(y)
        self.sv = x

        if self.avg_weight:
            w_avg = np.zeros(self.w.shape)
            rw_avg = np.zeros(self.rw.shape)

        # self.X_ = np.zeros([self.max_size, X.shape[1]])
        # self.rX_ = np.zeros([self.max_size, 2*self.D])

        mistake = 0.0

        # initialize mapping matrix for random features
        self.u = (2 * self.gamma) * np.random.randn(x.shape[1], self.D)

        # pre-allocate (FASTEST)
        rx = np.zeros([x.shape[0], 2 * self.D])
        rx[:, :self.D] = np.cos(x.dot(self.u)) / np.sqrt(self.D)
        rx[:, self.D:] = np.sin(x.dot(self.u)) / np.sqrt(self.D)

        # horizontal stack
        # rX = np.hstack([np.cos(X.dot(self.u_))/np.sqrt(self.D), np.sin(X.dot(self.u_))/np.sqrt(self.D)])

        # pre-allocate + sin-cos
        # rX = np.zeros([X.shape[0], 2*self.D])
        # sinx = np.sin(X.dot(self.u_))
        # cosx = np.abs((1-sinx**2)**0.5)
        # signx = np.sign(((X.dot(self.u_)-np.pi/2)%(2*np.pi))-np.pi)
        # rX[:, :self.D] = (cosx*signx) / np.sqrt(self.D)
        # rX[:, self.D:] = sinx / np.sqrt(self.D)

        for t in range(x.shape[0]):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            wx = self.get_wx(x, x[t], rx[t])
            if self.task == 'classification':
                if self.num_classes == 2:
                    y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                else:
                    y_pred = self._decode_labels(np.argmax(wx))
                mistake += (y_pred != y0[t])
            else:
                mistake += (wx[0] - y0[t]) ** 2
            alpha_t, z = self.get_grad(x, x[t], rx[t], y[t], wx=wx)  # compute \alpha_t

            # batch mode
            # alpha_t, z = self.get_grad(x, x[t], rx[t], y[t])  # compute \alpha_t

            self.w *= (1.0 * t) / (t + 1)
            self.rw *= (1.0 * t) / (t + 1)

            w = -alpha_t / (self.lbd * (t + 1))

            if self.budget_size == self.max_budget_size:
                self.maintain_budget(rx, w)
            self.add_to_core_set(t, w, y=y[t], z=z)

            if self.avg_weight:
                w_avg += self.w
                rw_avg += self.rw

            if self.record > 0 and (not ((t + 1) % self.record)):
                self.list_train_time.append(time.time() - self.start_time)
                self.list_mistake.append(mistake / (t + 1))
                self.list_mistake_idx.append(t + 1)

            epoch_logs.update({'model_size': self.budget_size,
                               'elapsed_time': time.time() - self.start_time})
            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

        if self.avg_weight:
            self.w = w_avg / x.shape[0]
            self.rw = rw_avg / x.shape[0]

        self.mistake = mistake / x.shape[0]

    def predict(self, x):
        y = np.zeros(x.shape[0])

        rx = np.zeros([x.shape[0], 2 * self.D])
        rx[:, :self.D] = np.cos(x.dot(self.u)) / np.sqrt(self.D)
        rx[:, self.D:] = np.sin(x.dot(self.u)) / np.sqrt(self.D)

        for i in range(x.shape[0]):
            wx = self.get_wx(self.sv, x[i], rx[i])
            if self.task == 'classification':
                if self.num_classes == 2:
                    y[i] = self._decode_labels(np.uint8(wx >= 0))
                else:
                    y[i] = self._decode_labels(np.argmax(wx))
            else:
                y[i] = wx[0]
        return y

    def score(self, x, y, sample_weight=None):
        return -INF if self.exception else -self.mistake

    def get_params(self, deep=True):
        out = super(DualSGD, self).get_params(deep=deep)
        param_names = DualSGD._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
