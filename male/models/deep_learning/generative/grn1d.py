from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from scipy.stats import norm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
config.allow_soft_placement = True

from male.models.distribution import Uniform1D
from male.models.distribution import Gaussian1D
from male.models.deep_learning.generative import GAN1D


class GRN1D(GAN1D):
    """Generative Reguarlized Nets for 1D data
    """

    def __init__(self,
                 model_name="GRN1D",
                 aux_coeffs=None,
                 aux_batch_size=2,
                 aux_discriminators=None,
                 aux_learning_rate=0.001,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(GRN1D, self).__init__(**kwargs)
        self.aux_coeffs = aux_coeffs
        self.aux_batch_size = aux_batch_size
        self.aux_discriminators = aux_discriminators
        self.aux_learning_rate = aux_learning_rate

    def _init(self):
        super(GRN1D, self)._init()

        if self.aux_discriminators is None:
            self.aux_discriminators = [Uniform1D(low=-5, high=-1), Uniform1D(low=1, high=5)]
        self.num_aux_ = len(self.aux_discriminators)  # number of auxiliary distributions
        if self.aux_coeffs is None:
            self.aux_coeffs = [0.1] * self.num_aux_

        # The auxiliary distributions
        self.a1_ = [None] * self.num_aux_
        self.a2_ = [None] * self.num_aux_
        self.ax_ = [None] * self.num_aux_
        for i in range(self.num_aux_):
            with tf.variable_scope('aux' + str(i + 1)) as scope:
                self.ax_[i] = tf.placeholder(tf.float32, shape=[None, 1])
                self.a1_[i] = self._create_discriminator(self.ax_[i], self.hidden_size)
                scope.reuse_variables()
                self.a2_[i] = self._create_discriminator(self.g_, self.hidden_size)

        # Add to the losses for generator networks
        for i in range(self.num_aux_):
            self.g_loss_ += self.aux_coeffs[i] * tf.reduce_mean(tf.log(self.a2_[i]))
        # Define the losses for auxiliary discriminators
        self.a_loss_ = [None] * self.num_aux_
        for i in range(self.num_aux_):
            self.a_loss_[i] = tf.reduce_mean(-tf.log(self.a1_[i]) - tf.log(1 - self.a2_[i]))

        self.a_params_ = [None] * self.num_aux_
        for i in range(self.num_aux_):
            self.a_params_[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=('aux' + str(i + 1)))

        self.g_opt_ = self._create_optimizer(self.g_loss_, self.g_params_,
                                             self.generator_learning_rate)
        self.a_opt_ = [None] * self.num_aux_
        for i in range(self.num_aux_):
            self.a_opt_[i] = self._create_optimizer(self.a_loss_[i], self.a_params_[i],
                                                    self.aux_learning_rate)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            while self.epoch_ < self.num_epochs:
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch_)

                batch_logs = {'batch': 0,
                              'size': self.batch_size}
                callbacks.on_batch_begin(0, batch_logs)

                # update discriminator
                x = self.data.sample(self.batch_size)
                x.sort()
                z = self.generator.stratified_sample(self.batch_size)
                d_loss, _ = sess.run([self.d_loss_, self.d_opt_],
                                     feed_dict={self.x_: np.reshape(x, [self.batch_size, 1]),
                                                self.z_: np.reshape(z, [self.batch_size, 1])})

                # update auxiliaries
                a_loss = [0] * self.num_aux_
                for i in range(self.num_aux_):
                    x = self.aux_discriminators[i].sample(self.aux_batch_size)
                    x.sort()
                    z = self.generator.stratified_sample(self.aux_batch_size)
                    a_loss[i], _ = sess.run(
                        [self.a_loss_[i], self.a_opt_[i]],
                        feed_dict={self.ax_[i]: np.reshape(x, [self.aux_batch_size, 1]),
                                   self.z_: np.reshape(z, [self.aux_batch_size, 1])}
                    )

                # update generator
                z = self.generator.stratified_sample(self.batch_size)
                g_loss, _ = sess.run([self.g_loss_, self.g_opt_],
                                     feed_dict={self.z_: np.reshape(z, [self.batch_size, 1])})

                batch_logs.update(self._on_batch_end(x))
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss
                for (i, l) in enumerate(a_loss):
                    batch_logs['a_loss_' + str(i + 1)] = l

                callbacks.on_batch_end(0, batch_logs)

                epoch_logs['d_loss'] = d_loss
                epoch_logs['g_loss'] = g_loss
                for (i, l) in enumerate(a_loss):
                    epoch_logs['a_loss_' + str(i + 1)] = l
                epoch_logs.update(self._on_epoch_end(sess=sess))
                callbacks.on_epoch_end(self.epoch_, epoch_logs)

                self.epoch_ += 1
                if self.stop_training_:
                    self.epoch_ = self.stop_training_
                    break

    def get_params(self, deep=True):
        out = super(GAN1D, self).get_params(deep=deep)
        out.update({
            'aux_coeffs': self.aux_coeffs,
            'aux_batch_size': self.aux_batch_size,
            'aux_learning_rate': self.aux_learning_rate,
            'aux_discriminators': self.aux_discriminators,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(GAN1D, self).get_all_params(deep=deep)
        out.update({})
        return out
