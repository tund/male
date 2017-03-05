from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import tensorflow as tf
from scipy.misc import logsumexp
from .tensorflow_rbm import TensorFlowRBM
from ....utils.func_utils import tf_logsumone

EPSILON = np.finfo(np.float32).eps


class BernoulliBernoulliTensorFlowRBM(TensorFlowRBM):
    def __init__(self, model_name="TensorFlow_BBRBM", **kwargs):
        kwargs["model_name"] = model_name
        super(BernoulliBernoulliTensorFlowRBM, self).__init__(**kwargs)

    def _init(self):
        super(BernoulliBernoulliTensorFlowRBM, self)._init()

    def _get_hidden_prob(self, vsample, **kwargs):
        return tf.nn.sigmoid(tf.matmul(vsample, self.w_) + self.h_)

    def _sample_hidden(self, hprob):
        return tf.to_float(tf.less(tf.random_uniform(hprob.get_shape()), hprob))

    def _get_visible_prob(self, hsample):
        return tf.nn.sigmoid(tf.matmul(hsample, tf.transpose(self.w_)) + self.v_)

    def _sample_visible(self, vprob):
        return tf.to_float(tf.less(tf.random_uniform(vprob.get_shape()), vprob))

    def _create_free_energy(self, x):
        wx = tf.matmul(x, self.w_) + self.h_
        return - tf.matmul(x, tf.transpose(self.v_)) \
               - tf.reduce_sum(tf_logsumone(wx), axis=1, keep_dims=True)
        # return - x.dot(self.v_.T) - logsumone(wx).sum(axis=1, keepdims=True)
        # return - x.dot(self.v_.T) - np.logaddexp(np.zeros(wx.shape), wx).sum(axis=1, keepdims=True)

    def get_conditional_loglik(self, x, hsample):
        vprob = np.clip(self._get_visible_prob(hsample), EPSILON, 1 - EPSILON)
        return logsumexp(x.dot(np.log(vprob.T)) + (1 - x).dot(np.log(1 - vprob.T)),
                         axis=1) - np.log(hsample.shape[0])

    def _create_reconstruction_loglik(self, x, rdata=None):
        rdata = self._create_reconstruction(x) if rdata is None else rdata
        rdata = tf.clip_by_value(rdata, EPSILON, 1 - EPSILON)
        return x * tf.log(rdata) + (1 - x) * tf.log(1 - rdata)

    def get_logpartition(self, method='exact'):
        if method == 'exact':
            if min(self.num_hidden, self.num_visible) > 20:
                raise ValueError(
                    "The BernoulliBernoulliRBM is too big to compute log-partition function exactly.")
            if self.num_hidden < self.num_visible:
                hsample = np.zeros((2 ** self.num_hidden, self.num_hidden))
                for i in range(2 ** self.num_hidden):
                    hsample[i] = [int(j) for j in list("{0:b}".format(i).zfill(self.num_hidden))]
                log_hprob = (hsample.dot(self.h_.T)
                             + logsumone(self.v_ + hsample.dot(self.w_.T)).sum(axis=1,
                                                                               keepdims=True))
                return logsumexp(log_hprob)
            else:
                vsample = np.zeros((2 ** self.num_visible, self.num_visible))
                for i in range(2 ** self.num_visible):
                    vsample[i] = [int(j) for j in list("{0:b}".format(i).zfill(self.num_visible))]
                log_vprob = (vsample.dot(self.v_.T)
                             + logsumone(self.h_ + vsample.dot(self.w_)).sum(axis=1, keepdims=True))
                return logsumexp(log_vprob)
        else:
            raise NotImplementedError

    def transform(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        hprob = sess.run(self.hidden_prob_, feed_dict={self.x_: x})
        if sess != self.tf_session_:
            sess.close()
        return hprob
