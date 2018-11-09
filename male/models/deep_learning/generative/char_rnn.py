from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from ....model import Model
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class CharRNN(Model):
    def __init__(self,
                 model_name="char_rnn",
                 hidden_size=100,
                 seq_length=25,
                 learning_rate=0.01,
                 num_epochs=50000,
                 vocab_size=65,
                 xh_init=0.01, hh_init=0.01, hy_init=0.01,
                 char_to_ix=None, ix_to_char=None,
                 **kwargs):

        super(CharRNN, self).__init__(model_name=model_name, **kwargs)

        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.vocab_size = vocab_size

        self.ix_to_char = ix_to_char
        self.char_to_ix = char_to_ix
        self.xh_init, self.hh_init, self.hy_init = xh_init, hh_init, hy_init

    def _init_params(self, x):
        # initialize parameters
        self.xh = np.random.randn(self.hidden_size, self.vocab_size) * self.xh_init  # input to hidden
        self.hh = np.random.randn(self.hidden_size, self.hidden_size) * self.hh_init  # hidden to hidden
        self.hy = np.random.randn(self.vocab_size, self.hidden_size) * self.hy_init  # hidden to output
        self.bh = np.zeros((self.hidden_size, 1))  # hidden bias
        self.by = np.zeros((self.vocab_size, 1))  # output bias
        self.h_prev = None

        return self.xh, self.hh, self.hy, self.bh, self.by, self.h_prev

    def _initialize_grad(self):
        # initialize parameters for gradients
        self.dxh, self.dhh, self.dhy = np.zeros_like(self.xh ), np.zeros_like(self.hh), np.zeros_like(self.hy)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        return self.dxh, self.dhh, self.dhy, self.dbh, self.dby

    def loss_function(self, inputs, targets, h_prev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        # record each hidden state of
        hs[-1] = np.copy(h_prev)
        loss = 0
        # forward pass for each training data point
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1

            # hidden state, using previous hidden state hs[t-1]
            hs[t] = np.tanh(np.dot(self.xh, xs[t]) + np.dot(self.hh, hs[t - 1]) + self.bh)
            # unnormalized log probabilities for next chars
            ys[t] = np.dot(self.hy, hs[t]) + self.by
            # probabilities for next chars, softmax
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            # softmax (cross-entropy loss)
            loss += -np.log(ps[t][targets[t], 0])

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy, dbh, dby = self._initialize_grad()
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            # compute derivative of error w.r.t the output probabilites
            # dE/dy[j] = y[j] - t[j]
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y

            # output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
            # of output layer.
            # then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
            # dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
            dWhy += np.dot(dy, hs[t].T)
            dby += dy

            # backprop into h
            # derivative of error with regard to the output of hidden layer
            # derivative of H, come from output layer y and also come from H(t+1), the next time H
            dh = np.dot(self.hy.T, dy) + dhnext
            # backprop through tanh nonlinearity
            # derivative of error with regard to the input of hidden layer
            # dtanh(x)/dx = 1 - tanh(x) * tanh(x)
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw

            # derivative of the error with regard to the weight between input layer and hidden layer
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            # derivative of the error with regard to H(t+1)
            # or derivative of the error of H(t-1) with regard to H(t)
            dhnext = np.dot(self.hh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def fit(self, x=None, y=None):
        self._init_params(x)
        self._fit_loop(x, y)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        # iterator counter
        self.epoch = 0
        # data pointer
        p = 0
        # initialize parameters of model
        mxh, mhh, mhy = np.zeros_like(self.xh), np.zeros_like(self.hh), np.zeros_like(self.hy)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0

        while self.epoch <= self.num_epochs:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.seq_length + 1 >= len(x) or self.epoch == 0:
                # reset RNN memory
                # hprev is the hiddden state of RNN
                self.h_prev = np.zeros((self.hidden_size, 1))
                # go from start of data
                p = 0

            inputs = [self.char_to_ix[ch] for ch in x[p: p + self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in x[p + 1: p + self.seq_length + 1]]

            loss, dxh, dhh, dhy, dbh, dby, self.h_prev = self.loss_function(inputs, targets, self.h_prev)

            # author using Adagrad(a kind of gradient descent)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if self.epoch % 100 == 0:
                print('iter %d, loss: %f' % (self.epoch, smooth_loss))  # print progress

            for param, dparam, mem in zip([self.xh, self.hh, self.hy, self.bh, self.by],
                                          [self.dxh, self.dhh, self.dhy, self.dbh, self.dby],
                                          [mxh, mhh, mhy, mbh, mby]):
                mem += dparam * dparam
                # learning_rate is adjusted by mem, if mem is getting bigger, then learning_rate will be small
                # gradient descent of Adagrad
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            self.epoch += 1
            p += self.seq_length

    # given a hidden RNN state, and a input char id, predict the coming n chars
    def _sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """

        # a one-hot vector
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1

        ixes = []
        for t in range(n):
            # self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
            h = np.tanh(np.dot(self.xh, x) + np.dot(self.hh, h) + self.bh)
            # y = np.dot(self.W_hy, self.h)
            y = np.dot(self.hy, h) + self.by
            # softmax
            p = np.exp(y) / np.sum(np.exp(y))
            # sample according to probability distribution
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())

            # update input x
            # use the new sampled result as last input, then predict next char again.
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1

            ixes.append(ix)

        return ixes

    def sample(self, n_length=200, seed="A"):
        sample_ix = self._sample(self.h_prev, self.char_to_ix[seed], n_length)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        return txt

    def get_params(self, deep=True):
        out = super(CharRNN, self).get_params(deep=deep)
        param_names = CharRNN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
