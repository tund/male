from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

np.seterr(all='raise')

from . import KSGD
from ...utils.generic_utils import make_batches

INF = 1e+8


class RKS(KSGD):
    """Random Kitchen Sinks
    """

    def __init__(self,
                 model_name="RKS",
                 D=100,  # number of random features
                 learning_rate=0.005,
                 **kwargs):
        super(RKS, self).__init__(model_name=model_name, **kwargs)
        self.learning_rate = learning_rate
        self.D = D

    def _init(self):
        super(RKS, self)._init()
        self.omega = None

    def _init_params(self, x):
        if self.num_classes > 2:
            self.w = 0.01 * self.random_engine.randn(2 * self.D, self.num_classes)
        else:
            self.w = 0.01 * self.random_engine.randn(2 * self.D)
        self.omega = self.gamma * self.random_engine.randn(x.shape[1], self.D)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        batches = make_batches(x.shape[0], self.batch_size)

        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                dw = self.get_grad(x_batch, y_batch)

                self.w -= self.learning_rate * dw

                batch_logs.update(self._on_batch_end(x_batch, y_batch))
                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def predict(self, x):
        if x.ndim < 2:
            x = x.copy()[..., np.newaxis]

        y = np.ones(x.shape[0])
        batches = make_batches(x.shape[0], self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x_batch = x[batch_start:batch_end]
            phi = self._get_phi(x_batch)
            wx = phi.dot(self.w)
            if self.task == 'classification':
                if self.num_classes == 2:
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
            wx = phi.dot(self.w)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y[batch_start:batch_end] = np.uint8(wx >= 0)
                else:
                    y[batch_start:batch_end] = np.argmax(wx, axis=1)
            else:
                y[batch_start:batch_end] = wx
        return y

    def _roll_params(self):
        return np.concatenate([super(RKS, self)._roll_params(),
                               np.ravel(self.w.copy())])

    def _unroll_params(self, w):
        ww = super(RKS, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww], dtype=np.int)
        w_ = w[idx:idx + self.w.size].reshape(self.w.shape).copy()
        if len(ww) == 0:
            return w_
        else:
            return ww, w_

    def _get_phi(self, x, **kwargs):
        omega = kwargs['omega'] if 'omega' in kwargs else self.omega

        phi = np.zeros([x.shape[0], 2 * self.D])  # Mx2D
        xo = x.dot(omega)
        phi[:, :self.D] = np.cos(xo)
        phi[:, self.D:] = np.sin(xo)
        return phi

    def _get_wxy(self, wx, y):
        m = len(y)  # batch size
        idx = range(m)
        mask = np.ones([m, self.num_classes], np.bool)
        mask[idx, y] = False
        z = np.argmax(wx[mask].reshape([m, self.num_classes - 1]), axis=1)
        z += (z >= y)
        return wx[idx, y] - wx[idx, z], z

    def get_grad(self, x, y, *arg, **kwargs):
        m = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w  # 2DxC
        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x)  # Mx2D
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # MxC

        dw = self.lbd * w  # 2DxC
        if self.num_classes > 2:
            wxy, z = self._get_wxy(wx, y)
            wxy = wxy[:, np.newaxis]
            if self.loss == 'hinge':
                d = (wxy < 1) * phi  # Mx2D
            else:  # logit loss
                d = np.exp(-wxy - np.logaddexp(0, -wxy)) * phi
            for i in range(self.num_classes):
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
        w = kwargs['w'] if 'w' in kwargs else self.w
        phi = self._get_phi(x, **kwargs)
        wx = phi.dot(w)

        f = (self.lbd / 2) * np.sum(w * w)
        if self.num_classes > 2:
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

    def get_params(self, deep=True):
        out = super(RKS, self).get_params(deep=deep)
        param_names = RKS._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
