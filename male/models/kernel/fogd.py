from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

np.seterr(all='raise')

from scipy.optimize import check_grad

from . import RKS

INF = 1e+8


class FOGD(RKS):
    """Fourier Online Gradient Descent
    """

    def __init__(self,
                 model_name="FOGD",
                 **kwargs):
        super(FOGD, self).__init__(model_name=model_name, **kwargs)

    def _init(self):
        super(FOGD, self)._init()
        self.mistake = INF

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

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

            wx = phi.dot(self.w)  # (x,)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                else:
                    y_pred = self._decode_labels(np.argmax(wx))
                mistake += (y_pred != y0[t])
            else:
                mistake += (wx[0] - y0[t]) ** 2
            dw = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients

            # update parameters
            self.w -= self.learning_rate * dw

            if self.avg_weight:
                w_avg += self.w

        if self.avg_weight:
            self.w = w_avg / x.shape[0]

        self.mistake = mistake / x.shape[0]

    def score(self, x, y, sample_weight=None):
        return -INF if self.exception else -self.mistake

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
            self.w -= self.learning_rate * dw

        s /= x.shape[0]
        print("diff = %.8f" % s)
        return s

    def get_params(self, deep=True):
        out = super(FOGD, self).get_params(deep=deep)
        param_names = FOGD._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
