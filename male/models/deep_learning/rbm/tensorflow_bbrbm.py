from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import tensorflow as tf
from scipy.misc import logsumexp
from .bbrbm import BernoulliBernoulliRBM
from .tensorflow_rbm import TensorFlowRBM
from ....utils.func_utils import tf_logsumone, logsumone

EPSILON = np.finfo(np.float32).eps


class BernoulliBernoulliTensorFlowRBM(TensorFlowRBM, BernoulliBernoulliRBM):
    def __init__(self, model_name="TensorFlow_BBRBM", **kwargs):
        kwargs["model_name"] = model_name
        super(BernoulliBernoulliTensorFlowRBM, self).__init__(**kwargs)

    def _init(self):
        super(BernoulliBernoulliTensorFlowRBM, self)._init()

    # def _get_hidden_prob(self, vsample, **kwargs):
    def _create_hidden_prob(self, tf_vsample, **kwargs):
        return tf.nn.sigmoid(tf.matmul(tf_vsample, self.tf_w_) + self.tf_h_)

    # def _sample_hidden(self, hprob):
    def _create_sample_hidden(self, tf_hprob):
        return tf.to_float(tf.less(tf.random_uniform(tf_hprob.get_shape()), tf_hprob))


    # def _get_visible_prob(self, hsample):
    def _create_visible_prob(self, tf_hsample):
        return tf.nn.sigmoid(tf.matmul(tf_hsample, tf.transpose(self.tf_w_)) + self.tf_v_)

    # def _sample_visible(self, vprob):
    def _create_sample_visible(self, tf_vprob):
        return tf.to_float(tf.less(tf.random_uniform(tf_vprob.get_shape()), tf_vprob))

    def _create_free_energy(self, tf_x):
        tf_wx = tf.matmul(tf_x, self.tf_w_) + self.tf_h_
        return - tf.matmul(tf_x, tf.transpose(self.tf_v_)) \
               - tf.reduce_sum(tf_logsumone(tf_wx), axis=1, keep_dims=True)
        # return - x.dot(self.v_.T) - logsumone(wx).sum(axis=1, keepdims=True)
        # return - x.dot(self.v_.T) - np.logaddexp(np.zeros(wx.shape), wx).sum(axis=1, keepdims=True)

    def get_conditional_loglik(self, x, hsample):
        vprob = np.clip(self._get_visible_prob(hsample), EPSILON, 1 - EPSILON)
        return logsumexp(x.dot(np.log(vprob.T)) + (1 - x).dot(np.log(1 - vprob.T)),
                         axis=1) - np.log(hsample.shape[0])

    def _create_reconstruction_loglik(self, tf_x, tf_rdata=None):
        tf_rdata = self._create_reconstruction(tf_x) if tf_rdata is None else tf_rdata
        tf_rdata = tf.clip_by_value(tf_rdata, EPSILON, 1 - EPSILON)
        return tf_x * tf.log(tf_rdata) + (1 - tf_x) * tf.log(1 - tf_rdata)

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
        hprob = sess.run(self.tf_hidden_prob_, feed_dict={self.tf_x_: x})
        if sess != self.tf_session_:
            sess.close()
        return hprob
