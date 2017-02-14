from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

np.seterr(all='raise')

from scipy.optimize import check_grad

from . import FOGD
from ...utils.generic_utils import make_batches


class RRF(FOGD):
    """Reparameterized Random Features
    """

    def __init__(self,
                 model_name="RRF",
                 learning_rate_gamma=0.005,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(RRF, self).__init__(**kwargs)
        self.learning_rate_gamma = learning_rate_gamma

    def _init(self):
        super(RRF, self)._init()
        self.gamma_ = None
        self.e_ = None
        self.num_features_ = 0  # number of data features

    def _init_params(self, x):
        super(RRF, self)._init_params(x)
        self.num_features_ = x.shape[1]
        self.gamma_ = np.log(self.gamma) * np.ones(self.num_features_)  # Nx1
        self.e_ = self.random_state_.randn(self.num_features_, self.D)  # NxD (\epsilon ~ N(0, 1))

    def _get_wxy(self, wx, y):
        m = len(y)  # batch size
        idx = range(m)
        mask = np.ones([m, self.num_classes_], np.bool)
        mask[idx, y] = False
        z = np.argmax(wx[mask].reshape([m, self.num_classes_ - 1]), axis=1)
        z += (z >= y)
        return wx[idx, y] - wx[idx, z], z

    def get_gamma_grad(self, x, phi, dphi, *args, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)

        m = x.shape[0]  # batch size
        # gradient of \phi w.r.t \omega
        dpo = np.zeros([m, 2 * self.D, self.num_features_])  # (M,2D,N)
        coswx, sinwx = phi[:, :self.D], phi[:, self.D:]  # (M,D)

        # broadcasting
        # drw[:, :self.D, :] = -X[:, np.newaxis, :] * sinwx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # drw[:, self.D:, :] = X[:, np.newaxis, :] * coswx[:, :, np.newaxis] * self.eps_.T[np.newaxis, :, :]  # (M,D,N)
        # dlg = drw.reshape([M * 2 * self.D, self.N_]).T.dot(dr.reshape(M * 2 * self.D)) * np.exp(g)

        # einsum
        dpo[:, :self.D, :] = np.einsum("mn,md,nd->mdn", -x, sinwx, self.e_)  # (M,D,N)
        dpo[:, self.D:, :] = np.einsum("mn,md,nd->mdn", x, coswx, self.e_)  # (M,D,N)
        dlg = np.einsum("mdn,md->n", dpo, dphi) * np.exp(gamma)

        return dlg

    def get_grad(self, x, y, *args, **kwargs):
        m = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w_  # (2D,C)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (N,)
        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (M,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (M,C)

        dw = self.lbd * w  # (2D,C)
        dgamma = np.zeros(gamma.shape)  # (N,)
        if self.num_classes_ > 2:
            wxy, z = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (M,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, z[wxy < 1]].T  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], dphi, gamma=gamma) / m
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, z].T)  # (M,2D)
                dgamma += self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            for i in range(self.num_classes_):
                dw[:, i] += -d[y == i].sum(axis=0) / m
                dw[:, i] += d[z == i].sum(axis=0) / m
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / m
                dphi = -y[wxy < 1, np.newaxis] * w  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy < 1], phi[wxy < 1], dphi, gamma=gamma) / m
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,2D)
                dgamma = self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (M,2D)
                dgamma = self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (M,2D)
                dgamma += self.get_gamma_grad(x, phi, dphi, gamma=gamma) / m
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / m
                dphi = c * w  # (M,2D)
                dgamma += self.get_gamma_grad(x[wxy], phi[wxy], dphi, gamma=gamma) / m
        return dw, dgamma

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        if self.mode == 'online':  # online setting
            y0 = self._decode_labels(y)
            if self.avg_weight:
                w_avg = np.zeros(self.w_.shape)

            mistake = 0.0
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
                dw, dgamma = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients

                # update parameters
                self.w_ -= self.learning_rate * dw
                self.gamma_ -= self.learning_rate_gamma * dgamma

                # update the average of parameters
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

                    dw, dgamma = self.get_grad(x_batch, y_batch)

                    self.w_ -= self.learning_rate * dw
                    self.gamma_ -= self.learning_rate_gamma * dgamma

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

    def _get_phi(self, x, **kwargs):
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_
        omega = np.exp(gamma)[:, np.newaxis] * self.e_  # NxD

        phi = np.zeros([x.shape[0], 2 * self.D])  # Mx2D
        xo = x.dot(omega)
        phi[:, :self.D] = np.cos(xo)
        phi[:, self.D:] = np.sin(xo)
        return phi

    def _roll_params(self):
        return np.concatenate([super(RRF, self)._roll_params(),
                               np.ravel(self.gamma_.copy())])

    def _unroll_params(self, w):
        ww = super(RRF, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww])
        gamma = w[idx:idx + self.gamma_.size].reshape(self.gamma_.shape).copy()
        return ww + (gamma,)

    def get_loss(self, x, y, *args, **kwargs):
        if 'gamma' not in kwargs:
            kwargs['gamma'] = self.gamma_
        return super(RRF, self).get_loss(x, y, **kwargs)

    def _get_loss_check_grad(self, w, x, y):
        ww, gamma = self._unroll_params(w)
        return self.get_loss(x, y, w=ww, gamma=gamma)

    def _get_grad_check_grad(self, w, x, y):
        ww, gamma = self._unroll_params(w)
        dw, dgamma = self.get_grad(x, y, w=ww, gamma=gamma)
        return np.concatenate([np.ravel(dw), np.ravel(dgamma)])

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

            dw, dgamma = self.get_grad(x[[t]], y[[t]])
            self.w_ -= self.learning_rate * dw
            self.gamma_ -= self.learning_rate_gamma * dgamma

        print("diff = %.8f" % (s / x.shape[0]))

    def get_params(self, deep=True):
        out = super(RRF, self).get_params(deep=deep)
        out.update({
            'learning_rate_gamma': self.learning_rate_gamma,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(RRF, self).get_all_params(deep=deep)
        out.update({
            'gamma_': copy.deepcopy(self.gamma_),
            'e_': copy.deepcopy(self.e_),
            'num_features_': self.num_features_,
        })
        return out
