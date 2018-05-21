from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from . import WGAN
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import linear, conv2d
from ....backend.tensorflow_backend import adam_optimizer


class WGAN_GP(WGAN):
    """Improved Wasserstein Generative Adversarial Nets with Gradient Penalty (WGAN-GP)
    """

    def __init__(self,
                 model_name="WGAN_GP",
                 lbd=10.0,
                 learning_rate=0.0001,
                 **kwargs):
        super(WGAN_GP, self).__init__(model_name=model_name, **kwargs)
        self.lbd = lbd
        self.learning_rate = learning_rate

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, [None,
                                             self.img_size[0], self.img_size[1], self.img_size[2]],
                                name="real_data")
        self.z = tf.placeholder(tf.float32, [None, self.num_z], name='noise')

        # create generator G
        self.g = self._create_generator(self.z)

        # create sampler to generate samples
        self.sampler = self._create_generator(self.z, train=False, reuse=True)

        # create discriminator D
        self.dx = self._create_discriminator(self.x, name="critic")
        self.dg = self._create_discriminator(self.g, name="critic", reuse=True)

        # Gradient penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = self.g - self.x
        interpolates = self.x + (epsilon * differences)
        self.dxx = self._create_discriminator(interpolates, name="critic", reuse=True)
        gradients = tf.gradients(self.dxx, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = self.lbd * tf.reduce_mean((slopes - 1.) ** 2)

        # define loss functions
        self.d_loss = tf.add(tf.reduce_mean(self.dg) - tf.reduce_mean(self.dx),
                             gradient_penalty, name="d_loss")
        self.g_loss = tf.reduce_mean(-self.dg, name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = adam_optimizer(self.d_loss, self.learning_rate, beta1=0.5, params=d_params)
        self.g_opt = adam_optimizer(self.g_loss, self.learning_rate, beta1=0.5, params=g_params)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        num_data = x.shape[0] - x.shape[0] % self.batch_size
        callbacks._update_params({'num_samples': num_data})
        batches = make_batches(num_data, self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_size = batch_end - batch_start
                batch_logs = {'batch': batch_idx,
                              'size': batch_size}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]
                z_batch = self.z_prior.sample([batch_size, self.num_z]).astype(np.float32)

                # update discriminator D
                d_loss = 0.0
                for _ in range(5):
                    d_loss, _ = self.tf_session.run(
                        [self.d_loss, self.d_opt],
                        feed_dict={self.x: x_batch, self.z: z_batch})

                # update generator G
                g_loss, _ = self.tf_session.run([self.g_loss, self.g_opt],
                                                feed_dict={self.z: z_batch})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            self._on_epoch_end(epoch_logs, input_data={self.x: x_batch, self.z: z_batch})
            callbacks.on_epoch_end(self.epoch - 1, epoch_logs)

    def _create_discriminator(self, x, train=True, reuse=False, name="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h = lrelu(conv2d(x, self.num_dis_feature_maps, stddev=0.02, name="d_h0_conv"))
            for i in range(1, self.num_conv_layers):
                h = lrelu(conv2d(h, self.num_dis_feature_maps * (2 ** i),
                                 stddev=0.02, name="d_h{}_conv".format(i)))
            dim = h.get_shape()[1:].num_elements()
            d_out = linear(tf.reshape(h, [-1, dim]), 1,
                           stddev=0.02, scope="d_out")
        return d_out

    def get_params(self, deep=True):
        out = super(WGAN, self).get_params(deep=deep)
        param_names = WGAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
