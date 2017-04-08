from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from male.models.distribution import Uniform1D
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
                 **kwargs):
        super(GRN1D, self).__init__(model_name=model_name, **kwargs)
        self.aux_coeffs = aux_coeffs
        self.aux_batch_size = aux_batch_size
        self.aux_discriminators = aux_discriminators
        self.aux_learning_rate = aux_learning_rate

    def _init(self):
        super(GRN1D, self)._init()

        if self.aux_discriminators is None:
            self.aux_discriminators = [Uniform1D(low=-5, high=-1), Uniform1D(low=1, high=5)]
        self.num_aux = len(self.aux_discriminators)  # number of auxiliary distributions
        if self.aux_coeffs is None:
            self.aux_coeffs = [0.1] * self.num_aux

    def _build_model(self, x):
        super(GRN1D, self)._build_model(x)

        # The auxiliary distributions
        self.a1 = [None] * self.num_aux
        self.a2 = [None] * self.num_aux
        self.ax = [None] * self.num_aux
        for i in range(self.num_aux):
            with tf.variable_scope('aux' + str(i + 1)) as scope:
                self.ax[i] = tf.placeholder(tf.float32, shape=[None, 1])
                self.a1[i] = self._create_discriminator(self.ax[i], self.hidden_size)
                scope.reuse_variables()
                self.a2[i] = self._create_discriminator(self.g, self.hidden_size)

        # Add to the losses for generator networks
        for i in range(self.num_aux):
            self.g_loss += self.aux_coeffs[i] * tf.reduce_mean(tf.log(self.a2[i]))
        # Define the losses for auxiliary discriminators
        self.a_loss = [None] * self.num_aux
        for i in range(self.num_aux):
            self.a_loss[i] = tf.reduce_mean(-tf.log(self.a1[i]) - tf.log(1 - self.a2[i]))

        self.a_params = [None] * self.num_aux
        for i in range(self.num_aux):
            self.a_params[i] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope=('aux' + str(i + 1)))

        self.g_opt = self._create_optimizer(self.g_loss, self.g_params,
                                            self.generator_learning_rate)
        self.a_opt = [None] * self.num_aux
        for i in range(self.num_aux):
            self.a_opt[i] = self._create_optimizer(self.a_loss[i], self.a_params[i],
                                                   self.aux_learning_rate)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            batch_logs = {'batch': 0,
                          'size': self.batch_size}
            callbacks.on_batch_begin(0, batch_logs)

            # update discriminator
            x = self.data.sample(self.batch_size)
            x.sort()
            z = self.generator.stratified_sample(self.batch_size)
            d_loss, _ = self.tf_session.run([self.d_loss, self.d_opt],
                                            feed_dict={self.x: np.reshape(x, [self.batch_size, 1]),
                                                       self.z: np.reshape(z, [self.batch_size, 1])})

            # update auxiliaries
            a_loss = [0] * self.num_aux
            for i in range(self.num_aux):
                x = self.aux_discriminators[i].sample(self.aux_batch_size)
                x.sort()
                z = self.generator.stratified_sample(self.aux_batch_size)
                a_loss[i], _ = self.tf_session.run(
                    [self.a_loss[i], self.a_opt[i]],
                    feed_dict={self.ax[i]: np.reshape(x, [self.aux_batch_size, 1]),
                               self.z: np.reshape(z, [self.aux_batch_size, 1])}
                )

            # update generator
            z = self.generator.stratified_sample(self.batch_size)
            g_loss, _ = self.tf_session.run([self.g_loss, self.g_opt],
                                            feed_dict={self.z: np.reshape(z, [self.batch_size, 1])})

            batch_logs.update(self._on_batch_end(x))
            batch_logs['d_loss'] = d_loss
            batch_logs['g_loss'] = g_loss
            for (i, l) in enumerate(a_loss):
                batch_logs['a_loss_' + str(i + 1)] = l
            batch_logs.update(self._on_batch_end(x))
            callbacks.on_batch_end(0, batch_logs)

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def get_params(self, deep=True):
        out = super(GRN1D, self).get_params(deep=deep)
        param_names = GRN1D._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
