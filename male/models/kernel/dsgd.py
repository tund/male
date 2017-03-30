from __future__ import absolute_import, division, print_function

import time

import numpy as np
from numpy.linalg import norm

from .sgd import KSGD

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
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(DualSGD, self).__init__(**kwargs)
        self.k = k
        self.D = D
        self.max_budget_size = max_budget_size
        self.maintain = maintain
        self.record = record

    def _init(self):
        super(DualSGD, self)._init()

        try:
            self.maintain = MAINTENANCE[self.maintain]
        except KeyError:
            raise ValueError("Maintenance strategy %s is not supported." % self.maintain)

        self.budget_size_ = 0
        self.rw_ = None  # weights for random features
        self.idx_ = None

        self.list_train_time_ = []
        self.list_score_ = []
        self.list_score_idx_ = []
        self.last_score_ = 0.0

    def _init_params(self, x):
        if self.num_classes_ > 2:
            self.w_ = np.zeros([self.max_budget_size, self.num_classes_])
            self.rw_ = np.zeros([2 * self.D, self.num_classes_])
        else:
            self.w_ = np.zeros([self.max_budget_size, 1])
            self.rw_ = np.zeros([2 * self.D, 1])
        self.idx_ = np.zeros(self.max_budget_size, dtype=np.int)

    def get_wx(self, x, x0, rx):
        b = self.budget_size_
        if b == 0:
            return np.array([0])
        else:
            if self.kernel == 'gaussian':
                xx = (x[self.idx_[:b]] - x0)
                return np.sum(self.w_[:b] * np.exp(-self.gamma * (xx * xx).sum(axis=1, keepdims=True)),
                              axis=0) + rx.dot(self.rw_)
            else:
                return np.array([0])

    def get_wxy(self, x, x0, rx, y, wx=None):
        if self.budget_size_ == 0:
            return (0, -1)
        else:
            if self.kernel == 'gaussian':
                if wx is None:
                    wx = self.get_wx(x, x0, rx)
                idx = np.ones(self.num_classes_, np.bool)
                idx[y] = False
                z = np.argmax(wx[idx])
                z += (z >= y)
                return (wx[y] - wx[z], z)
            else:
                return (0, -1)

    def get_grad(self, x, x0, rx, y, wx=None):
        if self.num_classes_ > 2:
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
        b = self.budget_size_
        self.idx_[b] = t
        if self.num_classes_ > 2:
            self.w_[b, y] = w
            if z >= 0:
                self.w_[b, z] = -w
        else:
            self.w_[b] = w
        self.budget_size_ += 1

    def remove(self, idx):
        n = len(idx)
        mask = np.ones(self.max_budget_size, np.bool)
        mask[idx] = False
        self.w_[:-n] = self.w_[mask]
        self.idx_[:-n] = self.idx_[mask]
        self.budget_size_ -= n
        # self.w_ = np.roll(self.w_, self.size_ - idx - 1)
        # self.idx_ = np.roll(self.idx_, self.size_ - idx - 1)
        # self.budget_size_ -= 1

    def maintain_budget(self, rx, w):
        if self.maintain == MAINTENANCE['k-merging']:
            i = np.argsort(norm(self.w_[:self.budget_size_], axis=1))
            # i = np.argpartition(norm(self.w_[:self.size_], axis=1), self.k)
            self.rw_ += rx[self.idx_[i[:self.k]]].T.dot(self.w_[i[:self.k]])
            self.remove(i[:self.k])

        elif self.maintain == MAINTENANCE['removal']:
            i = np.argmin(norm(self.w_, axis=1))
            mask = np.ones(self.max_budget_size, np.bool)
            mask[i] = False
            self.x_[:-1] = self.x_[mask]
            self.w_[:-1] = self.w_[mask]
            self.budget_size_ -= 1

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        y0 = self._decode_labels(y)

        if self.avg_weight:
            w_avg = np.zeros(self.w_.shape)
            rw_avg = np.zeros(self.rw_.shape)

        # self.X_ = np.zeros([self.max_size, X.shape[1]])
        # self.rX_ = np.zeros([self.max_size, 2*self.D])

        score = 0.0

        # initialize mapping matrix for random features
        self.u_ = (2 * self.gamma) * np.random.randn(x.shape[1], self.D)

        # pre-allocate (FASTEST)
        rx = np.zeros([x.shape[0], 2 * self.D])
        rx[:, :self.D] = np.cos(x.dot(self.u_)) / np.sqrt(self.D)
        rx[:, self.D:] = np.sin(x.dot(self.u_)) / np.sqrt(self.D)

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
            callbacks.on_epoch_begin(self.epoch_)

            wx = self.get_wx(x, x[t], rx[t])
            if self.task == 'classification':
                if self.num_classes_ == 2:
                    y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                else:
                    y_pred = self._decode_labels(np.argmax(wx))
                score += (y_pred != y0[t])
            else:
                score += (wx[0] - y0[t]) ** 2
            alpha_t, z = self.get_grad(x, x[t], rx[t], y[t], wx=wx)  # compute \alpha_t

            # batch mode
            # alpha_t, z = self.get_grad(x, x[t], rx[t], y[t])  # compute \alpha_t

            self.w_ *= (1.0 * t) / (t + 1)
            self.rw_ *= (1.0 * t) / (t + 1)

            w = -alpha_t / (self.lbd * (t + 1))

            if self.budget_size_ == self.max_budget_size:
                self.maintain_budget(rx, w)
            self.add_to_core_set(t, w, y=y[t], z=z)

            if self.avg_weight:
                w_avg += self.w_
                rw_avg += self.rw_

            if self.record > 0 and (not ((t + 1) % self.record)):
                self.list_train_time_.append(time.time() - self.start_time_)
                self.list_score_.append(score / (t + 1))
                self.list_score_idx_.append(t + 1)

            epoch_logs.update({'model_size': self.budget_size_,
                               'elapsed_time': time.time() - self.start_time_})
            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

        if self.avg_weight:
            self.w_ = w_avg / x.shape[0]
            self.rw_ = rw_avg / x.shape[0]

        self.last_score_ = score / x.shape[0]

    def predict(self, x):
        y = np.zeros(x.shape[0])

        rx = np.zeros([x.shape[0], 2 * self.D])
        rx[:, :self.D] = np.cos(x.dot(self.u_)) / np.sqrt(self.D)
        rx[:, self.D:] = np.sin(x.dot(self.u_)) / np.sqrt(self.D)

        for i in range(x.shape[0]):
            wx = self.get_wx(self.x_, x[i], rx[i])
            if self.task == 'classification':
                if self.num_classes_ == 2:
                    y[i] = self._decode_labels(np.uint8(wx[0] >= 0))
                else:
                    y[i] = self._decode_labels(np.argmax(wx))
            else:
                y[i] = wx[0]
        return y

    def score(self, x, y, sample_weight=None):
        return -self.last_score_

    '''
    def get_params(self, deep=True):
        out = super(DualSGD, self).get_params(deep=deep)
        out.update({
            'mode': self.mode,
            'max_budget_size': self.max_budget_size,
            'maintain': self.maintain,
            'k': self.k,
            'D': self.D,
            'record': self.record
        })
        return out

    def get_all_params(self, deep=True):
        out = super(KSGD, self).get_all_params(deep=deep)
        out.update({
            'budget_size_': self.budget_size_,
            'rw_': copy.deepcopy(self.rw_),
            'idx_': copy.deepcopy(self.idx_),
            'list_train_time_': copy.deepcopy(self.list_train_time_),
            'list_score_': copy.deepcopy(self.list_score_),
            'list_score_idx_': copy.deepcopy(self.list_score_idx_),
            'last_score_': self.last_score_
        })
        return out
    '''
