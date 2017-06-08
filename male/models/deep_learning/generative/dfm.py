from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import DCGAN
from ...distribution import Uniform1D
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....utils.generic_utils import conv_out_size_same
from ....backend.tensorflow_backend import linear, conv2d, deconv2d


class DFM(DCGAN):
    """Generative Adversarial Nets with Denoising Feature Matching
    """

    def __init__(self,
                 model_name='DFM',
                 alpha=0.03 / 1024,
                 noise_std=1.0,
                 num_dfm_layers=2,
                 num_dfm_hidden=1024,
                 **kwargs):
        super(DFM, self).__init__(model_name=model_name, **kwargs)
        self.alpha = alpha
        self.noise_std = noise_std
        self.num_dfm_layers = num_dfm_layers
        self.num_dfm_hidden = num_dfm_hidden

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, [None,
                                             self.img_size[0], self.img_size[1], self.img_size[2]],
                                name="real_data")
        self.z_prior = Uniform1D(low=-1.0, high=1.0)
        self.z = tf.placeholder(tf.float32, [None, self.num_z], name='noise')

        # create generator G
        self.g = self._create_generator(self.z)

        # create sampler to generate samples
        self.sampler = self._create_generator(self.z, train=False, reuse=True)

        # create discriminator D
        self.dx, dx_logits, dx_h = self._create_discriminator(self.x)
        self.dg, dg_logits, dg_h = self._create_discriminator(self.g, reuse=True)

        # create denoising autoencoder
        self.dax_loss = self._create_denoising_autoencoder(dx_h)
        self.dag_loss = self._create_denoising_autoencoder(dg_h, reuse=True)

        # define loss functions
        self.dx_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dx_logits, labels=tf.ones_like(self.dx)),
            name="dx_loss")
        self.dg_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dg_logits, labels=tf.zeros_like(self.dg)),
            name="dg_loss")
        self.d_loss = tf.add(self.dx_loss, self.dg_loss, name="d_loss")
        self.g_loss = tf.add(tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dg_logits,
                                                    labels=tf.ones_like(self.dg))),
            self.alpha * self.dag_loss, name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        da_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='denoising_autoencoder')
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.d_loss, var_list=d_params)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.g_loss, var_list=g_params)
        self.da_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.dax_loss, var_list=da_params)

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
                dx_loss, dg_loss, d_loss, _ = self.tf_session.run(
                    [self.dx_loss, self.dg_loss, self.d_loss, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch})

                # update denoising autoencoder
                da_loss, _ = self.tf_session.run(
                    [self.dax_loss, self.da_opt], feed_dict={self.x: x_batch})

                # update generator G
                g_loss, _ = self.tf_session.run([self.g_loss, self.g_opt],
                                                feed_dict={self.z: z_batch})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['dx_loss'] = dx_loss
                batch_logs['dg_loss'] = dg_loss
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            if (self.epoch + 1) % self.inception_score_freq == 0 and \
                            "inception_score" in self.metrics:
                epoch_logs['inception_score'] = self._compute_inception_score(
                    self.generate(num_samples=self.num_inception_samples))

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _create_generator(self, z, train=True, reuse=False, name="generator"):
        out_size = [(conv_out_size_same(self.img_size[0], 2),
                     conv_out_size_same(self.img_size[1], 2),
                     self.num_gen_feature_maps)]
        for i in range(self.num_conv_layers - 1):
            out_size = [(conv_out_size_same(out_size[0][0], 2),
                         conv_out_size_same(out_size[0][1], 2),
                         out_size[0][2] * 2)] + out_size

        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h0 = tf.nn.relu(batch_norm(linear(z, out_size[0][0] * out_size[0][1] * out_size[0][2],
                                              scope='g_h0_linear', stddev=0.02),
                                       decay=0.9,
                                       updates_collections=None,
                                       epsilon=1e-5,
                                       scale=True,
                                       is_training=train,
                                       scope="g_bn0"),
                            name="g_h0_relu")
            h = tf.reshape(h0, [-1, out_size[0][0], out_size[0][1], out_size[0][2]])

            for i in range(1, self.num_conv_layers):
                h = tf.nn.relu(
                    batch_norm(
                        deconv2d(h,
                                 [self.batch_size, out_size[i][0], out_size[i][1], out_size[i][2]],
                                 stddev=0.02, name="g_h{}_deconv".format(i)),
                        decay=0.9,
                        updates_collections=None,
                        epsilon=1e-5,
                        scale=True,
                        is_training=train,
                        scope="g_bn{}".format(i)),
                    name="g_h{}_relu".format(i))

            g_out = tf.nn.tanh(
                deconv2d(h, [self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]],
                         stddev=0.02, name="g_out_deconv"),
                name="g_out_tanh")

            return g_out

    def _create_discriminator(self, x, train=True, reuse=False, name="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h = lrelu(conv2d(x, self.num_dis_feature_maps, stddev=0.02, name="d_h0_conv"))
            for i in range(1, self.num_conv_layers):
                h = lrelu(batch_norm(conv2d(h, self.num_dis_feature_maps * (2 ** i),
                                            stddev=0.02, name="d_h{}_conv".format(i)),
                                     decay=0.9,
                                     updates_collections=None,
                                     epsilon=1e-5,
                                     scale=True,
                                     is_training=train,
                                     scope="d_bn{}".format(i)))
            dim = h.get_shape()[1:].num_elements()
            d_logits = linear(tf.reshape(h, [-1, dim]), 1,
                              stddev=0.02, scope="d_out_linear")
            d_out = tf.nn.sigmoid(d_logits, name="d_out_sigmoid")
        return d_out, d_logits, h

    def _create_denoising_autoencoder(self, x, reuse=False, name="denoising_autoencoder"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            dim = x.get_shape()[1:].num_elements()
            da_x = tf.reshape(x, [-1, dim])
            da_noise = tf.random_normal(shape=tf.shape(da_x), mean=0.0,
                                        stddev=self.noise_std, dtype=tf.float32)
            da_h = da_x + da_noise
            for i in range(self.num_dfm_layers):
                if i == 0:
                    da_h = lrelu(linear(da_h, self.num_dfm_hidden,
                                        scope="da_h{}".format(i), stddev=0.02))
                else:
                    da_h = lrelu(batch_norm(linear(da_h, self.num_dfm_hidden,
                                                   scope="da_h{}".format(i), stddev=0.02),
                                            decay=0.9,
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=True,
                                            scope="da_bn{}".format(i)))
            da_out = linear(da_h, da_x.get_shape()[1], scope="da_out", stddev=0.02)
            da_loss = tf.reduce_mean(tf.reduce_sum(tf.square(da_x - da_out), axis=1),
                                     name="da_loss")
            return da_loss

    def get_params(self, deep=True):
        out = super(DFM, self).get_params(deep=deep)
        param_names = DFM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
