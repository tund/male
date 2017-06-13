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
from ....utils.disp_utils import tile_raster_images
from ....backend.tensorflow_backend import linear, conv2d, deconv2d


class M2GAN(DCGAN):
    """Max-Margin Generative Adversarial Nets
    """

    def __init__(self,
                 model_name='M2GAN',
                 gamma_init=0.01,
                 train_gamma=True,
                 num_random_features=1000,
                 **kwargs):
        super(M2GAN, self).__init__(model_name=model_name, **kwargs)
        self.gamma_init = gamma_init
        self.train_gamma = train_gamma
        self.num_random_features = num_random_features

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
        self.dx = self._create_discriminator(self.x)
        self.dg = self._create_discriminator(self.g, reuse=True)

        # define loss functions
        self.dx_loss = tf.reduce_mean(tf.maximum(0.0, -self.dx), name="dx_loss")
        self.dg_loss = tf.reduce_mean(tf.maximum(0.0, 1 + self.dg), name="dg_loss")
        self.d_loss = tf.add(self.dx_loss, self.dg_loss, name="d_loss")
        self.g_loss = tf.reduce_mean(tf.maximum(0.0, -self.dg), name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.d_loss, var_list=d_params)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.g_loss, var_list=g_params)

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
        out = super(M2GAN, self).get_params(deep=deep)
        param_names = M2GAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
