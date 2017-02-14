from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy

import numpy as np

np.seterr(all='raise')

from scipy.optimize import check_grad

from . import KSGD
from ...utils.generic_utils import make_batches

INF = 1e+8


class FOGD(KSGD):
    """Fourier Online Gradient Descent
    """

    def __init__(self,
                 model_name="FOGD",
                 mode='batch',  # {'batch', 'online'}
                 D=100,  # number of random features
                 learning_rate=0.005,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(FOGD, self).__init__(**kwargs)
        self.mode = mode
        self.learning_rate = learning_rate
        self.D = D

    def _init(self):
        super(FOGD, self)._init()

        self.omega_ = None
        self.mistake_ = INF

    def _init_params(self, x):
        if self.num_classes_ > 2:
            self.w_ = 0.01 * self.random_state_.randn(2 * self.D, self.num_classes_)
        else:
            self.w_ = 0.01 * self.random_state_.randn(2 * self.D)
        self.omega_ = self.gamma * self.random_state_.randn(x.shape[1], self.D)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        if self.mode == 'online':  # online setting
            y0 = self._decode_labels(y)
            if self.avg_weight:
                w_avg = np.zeros(self.w_.shape)

            # self.X_ = np.zeros([self.max_size, X.shape[1]])
            # self.rX_ = np.zeros([self.max_size, 2*self.D])

            mistake = 0.0

            # initialize mapping matrix for random features
            # self.u_ = self.gamma * np.random.randn(X.shape[1], self.D)

            # pre-allocate (FASTEST)
            # rX = np.zeros([X.shape[0], 2 * self.D])
            # rX[:, :self.D] = np.cos(X.dot(self.u_))
            # rX[:, self.D:] = np.sin(X.dot(self.u_))

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
                phi = self._get_phi(x[[t]])

                wx = phi.dot(self.w_)  # (x,)
                if self.task == 'classification':
                    if self.num_classes_ == 2:
                        y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                    else:
                        y_pred = self._decode_labels(np.argmax(wx))
                    mistake += (y_pred != y0[t])
                else:
                    mistake += (wx[0] - y0[t]) ** 2
                dw = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients

                # update parameters
                self.w_ -= self.learning_rate * dw

                if self.avg_weight:
                    w_avg += self.w_

            if self.avg_weight:
                self.w_ = w_avg / x.shape[0]

            self.mistake_ = mistake / x.shape[0]
        else:  # batch setting
            batches = make_batches(x.shape[0], self.batch_size)

            while (self.epoch_ < self.num_epochs) and (not self.stop_training_):
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch_)

                for batch_idx, (batch_start, batch_end) in enumerate(batches):
                    batch_logs = {'batch': batch_idx,
                                  'size': batch_end - batch_start}
                    callbacks.on_batch_begin(batch_idx, batch_logs)

                    x_batch = x[batch_start:batch_end]
                    y_batch = y[batch_start:batch_end]

                    dw = self.get_grad(x_batch, y_batch)

                    self.w_ -= self.learning_rate * dw

                    outs = self._on_batch_end(x_batch, y_batch)
                    for l, o in zip(self.metrics, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_idx, batch_logs)

                if do_validation:
                    outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                    for l, o in zip(self.metrics, outs):
                        epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(self.epoch_, epoch_logs)
                self._on_epoch_end()

    def predict(self, x):
        if x.ndim < 2:
            x = x.copy()[..., np.newaxis]

        y = np.ones(x.shape[0])
        batches = make_batches(x.shape[0], self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x[batch_start:batch_end]
            phi = self._get_phi(x_batch)
            wx = phi.dot(self.w_)
            if self.task == 'classification':
                if self.num_classes_ == 2:
                    y[batch_start:batch_end] = self._decode_labels(np.uint8(wx >= 0))
                else:
                    y[batch_start:batch_end] = self._decode_labels(np.argmax(wx, axis=1))
            else:
                y[batch_start:batch_end] = wx
        return y

    def _predict(self, x):
        if x.ndim < 2:
            x = x.copy()[..., np.newaxis]

        y = np.ones(x.shape[0])
        batches = make_batches(x.shape[0], self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x[batch_start:batch_end]
            phi = self._get_phi(x_batch)
            wx = phi.dot(self.w_)
            if self.task == 'classification':
                if self.num_classes_ == 2:
                    y[batch_start:batch_end] = np.uint8(wx >= 0)
                else:
                    y[batch_start:batch_end] = np.argmax(wx, axis=1)
            else:
                y[batch_start:batch_end] = wx
        return y

    def score(self, x, y, sample_weight=None):
        if self.mode == 'online':
            return -self.mistake_
        else:
            return super(FOGD, self).score(x, y)

    def _roll_params(self):
        return np.concatenate([super(FOGD, self)._roll_params(),
                               np.ravel(self.w_.copy())])

    def _unroll_params(self, w):
        ww = super(FOGD, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww])
        w_ = w[idx:idx + self.w_.size].reshape(self.w_.shape).copy()
        if len(ww) == 0:
            return w_
        else:
            return ww, w_

    def _get_phi(self, x, **kwargs):
        omega = kwargs['omega'] if 'omega' in kwargs else self.omega_

        phi = np.zeros([x.shape[0], 2 * self.D])  # Mx2D
        xo = x.dot(omega)
        phi[:, :self.D] = np.cos(xo)
        phi[:, self.D:] = np.sin(xo)
        return phi

    def _get_wxy(self, wx, y):
        m = len(y)  # batch size
        idx = range(m)
        mask = np.ones([m, self.num_classes_], np.bool)
        mask[idx, y] = False
        z = np.argmax(wx[mask].reshape([m, self.num_classes_ - 1]), axis=1)
        z += (z >= y)
        return wx[idx, y] - wx[idx, z], z

    def get_grad(self, x, y, *arg, **kwargs):
        m = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w_  # 2DxC
        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x)  # Mx2D
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # MxC

        dw = self.lbd * w  # 2DxC
        if self.num_classes_ > 2:
            wxy, z = self._get_wxy(wx, y)
            wxy = wxy[:, np.newaxis]
            if self.loss == 'hinge':
                d = (wxy < 1) * phi  # Mx2D
            else:  # logit loss
                d = np.exp(-wxy - np.logaddexp(0, -wxy)) * phi
            for i in range(self.num_classes_):
                dw[:, i] += -d[y == i].sum(axis=0) / m
                dw[:, i] += d[z == i].sum(axis=0) / m
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / m
            elif self.loss == 'l1':
                dw += (np.sign(wx - y)[:, np.newaxis] * phi).mean(axis=0)
            elif self.loss == 'l2':
                dw += ((wx - y)[:, np.newaxis] * phi).mean(axis=0)
            elif self.loss == 'logit':
                wxy = y * wx
                dw += np.mean((-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis] * phi,
                              axis=0)
            elif self.loss == 'eps_insensitive':
                d = np.sign(wx - y)[:, np.newaxis] * phi
                dw += d[np.abs(y - wx) > self.eps].sum(axis=0) / m
        return dw

    def get_loss(self, x, y, *args, **kwargs):
        w = kwargs['w'] if 'w' in kwargs else self.w_
        phi = self._get_phi(x, **kwargs)
        wx = phi.dot(w)

        f = (self.lbd / 2) * np.sum(w * w)
        if self.num_classes_ > 2:
            wxy, z = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                f += np.maximum(0, 1 - wxy).mean()
            else:  # logit loss
                f += np.logaddexp(0, -wxy).mean()
        else:
            if self.loss == 'hinge':
                f += np.maximum(0, 1 - y * wx).mean()
            elif self.loss == 'l1':
                f += np.abs(y - wx).mean()
            elif self.loss == 'l2':
                f += np.mean(0.5 * ((y - wx) ** 2))
            elif self.loss == 'logit':
                f += np.logaddexp(0, -y * wx).mean()
            elif self.loss == 'eps_insensitive':
                f += np.maximum(0, np.abs(y - wx) - self.eps).mean()

        return f

    def _get_loss_check_grad(self, w, x, y):
        ww = self._unroll_params(w)
        return self.get_loss(x, y, w=ww)

    def _get_grad_check_grad(self, w, x, y):
        ww = self._unroll_params(w)
        dw = self.get_grad(x, y, w=ww)
        return np.ravel(dw)

    def check_grad_online(self, x, y):
        """Check gradients of the model using data X and label y if available
         """
        self._init()

        if y is not None:
            # encode labels
            y = self._encode_labels(y)

        # initialize weights
        self._init_params(x)

        print("Checking gradient... ", end='')

        s = 0.0
        for t in range(x.shape[0]):
            s += check_grad(self._get_loss_check_grad,
                            self._get_grad_check_grad,
                            self._roll_params(),
                            x[[t]], y[[t]])

            dw = self.get_grad(x[[t]], y[[t]])
            self.w_ -= self.learning_rate * dw

        print("diff = %.8f" % (s / x.shape[0]))

    def get_params(self, deep=True):
        out = super(FOGD, self).get_params(deep=deep)
        out.update({
            'D': self.D,
            'mode': self.mode,
            'learning_rate': self.learning_rate,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(FOGD, self).get_all_params(deep=deep)
        out.update({
            'omega_': copy.deepcopy(self.omega_),
            'mistake_': self.mistake_
        })
        return out
