from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ... import Model
from ...utils.generic_utils import make_batches

EPS = np.finfo(np.float32).eps


class MLP(Model):

    def __init__(self,
                 model_name='MLP',
                 loss='softmax',
                 hidden_units_list=(),
                 learning_rate=0.01,
                 reg_lambda=0.01,
                 **kwargs
                 ):

        super(MLP, self).__init__(model_name=model_name, **kwargs)
        self.loss = loss
        self.hidden_units_list = hidden_units_list
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def _init(self):
        super(MLP, self)._init()
        # initialize weights and biases and other members here
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.onehot_encoder = None

    def _init_params(self, x):
        # initialize weights
        n_hidden = self.hidden_units_list[0]
        self.w1 = self.random_engine.randn(x.shape[1], n_hidden) / np.sqrt(x.shape[1])
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = self.random_engine.randn(n_hidden, self.num_classes) / np.sqrt(n_hidden)
        self.b2 = np.zeros((1, self.num_classes))

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
                # current_batch_size = batch_end - batch_start

                dw1, db1, dw2, db2 = self.get_grad(x_batch, y_batch)

                # Gradient descent parameter update
                self.w1 += - self.learning_rate * dw1
                self.b1 += - self.learning_rate * db1
                self.w2 += - self.learning_rate * dw2
                self.b2 += - self.learning_rate * db2

                batch_logs.update(self._on_batch_end(x_batch, y_batch))
                callbacks.on_batch_end(batch_idx, batch_logs)

            # if self.epoch % 100 == 0:
            #     print("MLP loss after iteration %i: %f" % (self.epoch, self.get_loss(x, y)))

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def get_loss(self, x, y, *args, **kwargs):
        w1 = kwargs['w1'] if 'w1' in kwargs else self.w1
        b1 = kwargs['b1'] if 'b1' in kwargs else self.b1
        w2 = kwargs['w2'] if 'w2' in kwargs else self.w2
        b2 = kwargs['b2'] if 'b2' in kwargs else self.b2

        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Calculating the loss
        # TODO very bad performance I guess, need a way to work with encoded labels without decoding in each step
        correct_logprobs = -np.log(probs[range(x.shape[0]), np.argmax(y, axis=1)])
        data_loss = np.sum(correct_logprobs) / x.shape[0]

        # Add regularization term to loss
        data_loss += self.reg_lambda / 2.0 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return data_loss

    def get_grad(self, x, y, *args, **kwargs):
        w1 = kwargs['w1'] if 'w1' in kwargs else self.w1
        b1 = kwargs['b1'] if 'b1' in kwargs else self.b1
        w2 = kwargs['w2'] if 'w2' in kwargs else self.w2
        b2 = kwargs['b2'] if 'b2' in kwargs else self.b2

        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        # TODO very bad performance I guess, need a way to work with encoded labels without decoding in each step
        delta3[range(x.shape[0]), np.argmax(y, axis=1)] -= 1
        delta3 /= x.shape[0]

        dw2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms
        dw2 += self.reg_lambda * w2
        dw1 += self.reg_lambda * w1

        return dw1, db1, dw2, db2

    def _get_loss_check_grad(self, w, x, y):
        w1, b1, w2, b2 = self._unroll_params(w)
        return self.get_loss(x, y, w1=w1, b1=b1, w2=w2, b2=b2)

    def _get_grad_check_grad(self, w, x, y):
        w1, b1, w2, b2 = self._unroll_params(w)
        dw1, db1, dw2, db2 = self.get_grad(x, y, w1=w1, b1=b1, w2=w2, b2=b2)
        return np.concatenate([np.ravel(dw1), np.ravel(db1), np.ravel(dw2), np.ravel(db2)])

    def _encode_labels(self, y):
        yy = y.copy()
        yy = super(MLP, self)._encode_labels(yy)
        if self.loss == 'softmax':
            self.onehot_encoder = OneHotEncoder()
            yy = self.onehot_encoder.fit_transform(yy.reshape(-1, 1)).toarray()
        return yy

    def _decode_labels(self, y):
        yy = y.copy()
        if self.loss == 'softmax':
            yy = np.argmax(yy, axis=1)
        return super(MLP, self)._decode_labels(yy)

    def _transform_labels(self, y):
        yy = y.copy()
        yy = super(MLP, self)._transform_labels(yy)
        if self.loss == 'softmax':
            yy = self.onehot_encoder.transform(yy.reshape(-1, 1)).toarray()
        return yy

    def predict(self, x):
        check_is_fitted(self, "w1")
        check_is_fitted(self, "b1")
        check_is_fitted(self, "w2")
        check_is_fitted(self, "b2")

        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self._decode_labels(probs)

    def predict_2d_only(self, x):
        # Used only for prediciting with 2 features for 2D plotting

        check_is_fitted(self, "w1")
        check_is_fitted(self, "b1")
        check_is_fitted(self, "w2")
        check_is_fitted(self, "b2")

        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def _roll_params(self):
        return np.concatenate([super(MLP, self)._roll_params(),
                               np.ravel(self.w1.copy()),
                               np.ravel(self.b1.copy()),
                               np.ravel(self.w2.copy()),
                               np.ravel(self.b2.copy())])

    def _unroll_params(self, params):
        parent_params = super(MLP, self)._unroll_params(params)
        parent_params = tuple([parent_params]) if not isinstance(parent_params, tuple) else parent_params
        idx = np.sum([i.size for i in parent_params], dtype=np.int32)
        w1_ = params[idx:idx + self.w1.size].reshape(self.w1.shape).copy()
        idx += self.w1.size
        b1_ = params[idx:idx + self.b1.size].reshape(self.b1.shape).copy()
        idx += self.b1.size
        w2_ = params[idx:idx + self.w2.size].reshape(self.w2.shape).copy()
        idx += self.w2.size
        b2_ = params[idx:idx + self.b2.size].reshape(self.b2.shape).copy()
        return parent_params + (w1_, b1_, w2_, b2_)

    def get_params(self, deep=True):
        out = super(MLP, self).get_params(deep=deep)
        param_names = MLP._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
