from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from . import DCGAN
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import rmsprop_optimizer
from ....backend.tensorflow_backend import get_activation_summary
from ....backend.tensorflow_backend import linear, conv2d, batch_norm


class WGAN(DCGAN):
    """Wasserstein Generative Adversarial Nets (WGAN)
    """

    def __init__(self,
                 model_name="WGAN",
                 learning_rate=0.00005,
                 **kwargs):
        super(WGAN, self).__init__(model_name=model_name, **kwargs)
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
        self.dx = self._create_discriminator(self.x)
        self.dg = self._create_discriminator(self.g, reuse=True)

        # define loss functions
        self.d_loss = tf.subtract(tf.reduce_mean(self.dg),
                                  tf.reduce_mean(self.dx), name="d_loss")
        self.g_loss = tf.reduce_mean(-self.dg, name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = rmsprop_optimizer(self.d_loss, self.learning_rate, params=d_params)
        self.g_opt = rmsprop_optimizer(self.g_loss, self.learning_rate, params=g_params)
        # self.d_clipping = tf.group(dp.assign(tf.clip_by_value(dp, -0.01, 0.01)) for dp in d_params)
        with tf.control_dependencies(
                dp.assign(tf.clip_by_value(dp, -0.01, 0.01)) for dp in d_params):
            self.d_clipping = tf.no_op()

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
                    self.tf_session.run(self.d_clipping)

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
            get_activation_summary(h, "d_h0_lrelu")

            for i in range(1, self.num_conv_layers):
                h = lrelu(batch_norm(conv2d(h, self.num_dis_feature_maps * (2 ** i),
                                            stddev=0.02, name="d_h{}_conv".format(i)),
                                     is_training=train,
                                     scope="d_bn{}".format(i)))
                get_activation_summary(h, "d_h{}_lrelu".format(i))

            dim = h.get_shape()[1:].num_elements()
            d_out = linear(tf.reshape(h, [-1, dim]), 1,
                           stddev=0.02, scope="d_out")
            get_activation_summary(d_out, "d_out_linear")
        return d_out

    def get_params(self, deep=True):
        out = super(WGAN, self).get_params(deep=deep)
        param_names = WGAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
