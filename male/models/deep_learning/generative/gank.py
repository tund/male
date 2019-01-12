from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import DCGAN
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....utils.generic_utils import conv_out_size_same
from ....utils.disp_utils import tile_raster_images
from ....backend.tensorflow_backend import linear, conv2d, deconv2d


class GANK(DCGAN):
    """Generative Adversarial Nets with Kernels
    """

    def __init__(self,
                 model_name='GANK',
                 gamma_init=0.01,
                 train_gamma=True,
                 loss='logit',
                 num_random_features=1000,
                 **kwargs):
        super(GANK, self).__init__(model_name=model_name, **kwargs)
        self.gamma_init = gamma_init
        self.train_gamma = train_gamma
        self.loss = loss
        self.num_random_features = num_random_features

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
        self.dx_loss = tf.reduce_mean(
            self._loss(self.dx, tf.ones_like(self.dx)),
            name="dx_loss")
        self.dg_loss = tf.reduce_mean(
            self._loss(self.dg,
                       -tf.ones_like(self.dg) if self.loss == 'hinge' else tf.zeros_like(self.dg)),
            name="dg_loss")
        self.d_loss = tf.add(self.dx_loss, self.dg_loss, name="d_loss")
        self.g_loss = tf.reduce_mean(
            self._loss(self.dg, tf.ones_like(self.dg)),
            name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.d_loss, var_list=d_params)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.g_loss, var_list=g_params)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        num_data = x.shape[0] - x.shape[0] % self.batch_size
        callbacks._update_params({'num_samples': num_data})
        batches = make_batches(num_data, self.batch_size)
        iteration = 0
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_size = batch_end - batch_start
                if batch_size != self.batch_size:
                    continue

                batch_logs = {'batch': batch_idx,
                              'size': batch_size}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]
                z_batch = self.z_prior.sample([batch_size, self.num_z]).astype(np.float32)

                # update discriminator D
                dx_loss, dg_loss, d_loss, _ = self.tf_session.run(
                    [self.dx_loss, self.dg_loss, self.d_loss, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch})

                # update generator G
                g_loss, _ = self.tf_session.run([self.g_loss, self.g_opt],
                                                feed_dict={self.z: z_batch})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['dx_loss'] = dx_loss
                batch_logs['dg_loss'] = dg_loss
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

                if iteration % self.summary_freq == 0:
                    summary = self.tf_session.run(self.tf_merged_summaries,
                                                  feed_dict={self.x: x_batch, self.z: z_batch})
                    self.tf_summary_writer.add_summary(summary, iteration)

                iteration += 1

            self._on_epoch_end(epoch_logs, input_data={self.x: x_batch, self.z: z_batch})
            callbacks.on_epoch_end(self.epoch - 1, epoch_logs)

    def _loss(self, x, y):
        if self.loss == 'hinge':
            return tf.maximum(0.0, 1 - tf.multiply(x, y))
        elif self.loss == 'logit':
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        else:
            raise NotImplementedError

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
                tf.summary.histogram("d_lrelu", h)
            dim = h.get_shape()[1:].num_elements()
            h = tf.reshape(h, [-1, dim])

            # try the last layer with LReLU activation to test
            # phi_x = lrelu(linear(h, self.num_random_features, stddev=0.02, scope="d_last"))
            # tf.summary.histogram("phi_x", phi_x)

            log_gamma = tf.get_variable(name='log_gamma', shape=[1],
                                        initializer=tf.constant_initializer(
                                            np.log(self.gamma_init)),
                                        trainable=self.train_gamma)
            e = tf.get_variable(name="unit_noise", shape=[dim, self.num_random_features],
                                initializer=tf.random_normal_initializer(), trainable=False)
            omega = tf.multiply(tf.exp(log_gamma), e, name='omega')
            tf.summary.histogram("omega", omega)
            phi = tf.matmul(h, omega, name='phi')
            tf.summary.histogram("phi", phi)
            phi_x = tf.concat([tf.cos(phi) / np.sqrt(self.num_random_features),
                               tf.sin(phi) / np.sqrt(self.num_random_features)],
                              1, name='phi_x')
            tf.summary.histogram("phi_x", phi_x)

            d_out = linear(phi_x, 1, stddev=0.02, scope="d_out_linear")
            tf.summary.histogram("d_out", d_out)
        return d_out

    def generate(self, num_samples=100):
        sess = self._get_session()
        z = self.z_prior.sample([num_samples, self.num_z]).astype(np.float32)
        x = np.zeros([num_samples, self.img_size[0], self.img_size[1], self.img_size[2]],
                     dtype=np.float32)
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_size = batch_end - batch_start
            z_batch = z[batch_start:batch_end]
            if batch_size < self.batch_size:
                z_batch = np.vstack(
                    [z_batch, np.zeros([self.batch_size - batch_size,
                                        self.num_z])])
            x[batch_start:batch_end] = sess.run(
                self.sampler,
                feed_dict={self.z: z_batch})[:batch_size]
        if sess != self.tf_session:
            sess.close()
        return (x + 1.0) / 2.0

    def disp_generated_data(self, x, tile_shape=None,
                            output_pixel_vals=False, **kwargs):
        if tile_shape is None:
            tile_shape = (x.shape[0], 1)
        if self.img_size[2] == 1:
            img = tile_raster_images(x.reshape([x.shape[0], -1]),
                                     img_shape=(self.img_size[0], self.img_size[1]),
                                     tile_shape=tile_shape,
                                     tile_spacing=(1, 1),
                                     scale_rows_to_unit_interval=False,
                                     output_pixel_vals=output_pixel_vals)
        else:
            img = tile_raster_images((x[..., 0], x[..., 1], x[..., 2], None),
                                     img_shape=(self.img_size[0], self.img_size[1]),
                                     tile_shape=tile_shape,
                                     tile_spacing=(1, 1),
                                     scale_rows_to_unit_interval=False,
                                     output_pixel_vals=output_pixel_vals)
        if 'ax' in kwargs:
            ax = kwargs['ax']
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            ax.grid(0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if 'epoch' in kwargs:
                ax.set_xlabel("epoch #{}".format(kwargs['epoch']), fontsize=28)
        else:
            fig, ax = plt.subplots()
            ax.set_title(kwargs['title'] if 'title' in kwargs else "Learned weights",
                         fontsize=28)
            ax.axis('off')
            plt.colorbar()
            _ = ax.imshow(img, aspect='auto',
                          cmap=kwargs['color'] if 'color' in kwargs else 'Greys_r',
                          interpolation=kwargs[
                              'interpolation'] if 'interpolation' in kwargs else 'none')
            plt.show()

    def display(self, param, **kwargs):
        if param == 'x_samples':
            x = self.generate(num_samples=kwargs['num_samples']) if 'num_samples' in kwargs else \
                self.generate(num_samples=100)
            self.disp_generated_data(x, **kwargs)

    def create_image_grid(self, x, tile_shape=None, output_pixel_vals=False, **kwargs):
        if tile_shape is None:
            tile_shape = (x.shape[0], 1)
        if self.img_size[2] == 1:
            img = tile_raster_images(x.reshape([x.shape[0], -1]),
                                     img_shape=(self.img_size[0], self.img_size[1]),
                                     tile_shape=tile_shape,
                                     tile_spacing=(1, 1),
                                     scale_rows_to_unit_interval=False,
                                     output_pixel_vals=output_pixel_vals)
        else:
            img = tile_raster_images((x[..., 0], x[..., 1], x[..., 2], None),
                                     img_shape=(self.img_size[0], self.img_size[1]),
                                     tile_shape=tile_shape,
                                     tile_spacing=(1, 1),
                                     scale_rows_to_unit_interval=False,
                                     output_pixel_vals=output_pixel_vals)
        return img

    def generate_images(self, param, **kwargs):
        if param == 'x_samples':
            x = self.generate(num_samples=kwargs['num_samples']) if 'num_samples' in kwargs else \
                self.generate(num_samples=100)
            return self.create_image_grid(x, **kwargs)

    def get_params(self, deep=True):
        out = super(GANK, self).get_params(deep=deep)
        param_names = GANK._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
