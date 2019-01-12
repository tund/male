from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from functools import partial
from ....utils.generic_utils import int2tuple
from ....backend.tensorflow.ops import he_initializer, xavier_initializer
from ....backend.tensorflow.ops import batch_norm, get_activation_summary
from ....backend.tensorflow.layers import linear, conv2d, residual_block, attention


class ResnetGenerator(object):
    def __init__(self, feature_maps=128, img_size=(32, 32, 3), blocks=('up', 'up', 'up'),
                 kernel_initializer=he_initializer,
                 skip_initializer=xavier_initializer,
                 use_batch_norm=True,
                 use_spectral_norm=False,
                 l2_reg=None,
                 name='generator', **kwargs):
        self.feature_maps = int2tuple(feature_maps, reps=len(blocks) + 1)
        self.blocks = blocks
        self.bottom_height = (img_size[0] >> blocks.count('up')) << blocks.count('down')
        self.bottom_width = (img_size[1] >> blocks.count('up')) << blocks.count('down')
        self.num_img_channels = img_size[2]
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.kernel_initializer = kernel_initializer
        self.skip_initializer = skip_initializer
        self.l2_reg = l2_reg
        self.name = name

    def __call__(self, z, training=True, reuse=False, update_spectral_norm=True, **kwargs):
        use_spectral_norm = self.use_spectral_norm
        if not isinstance(self.use_batch_norm, bool):
            batch_norm_func = self.use_batch_norm
        else:
            batch_norm_func = partial(batch_norm,
                                      training=training) if self.use_batch_norm else None

        with tf.variable_scope(self.name, reuse=reuse):
            h = linear(z,
                       self.bottom_height * self.bottom_width * self.feature_maps[0],
                       use_spectral_norm=use_spectral_norm,
                       l2_reg=self.l2_reg,
                       name='g.h0.linear')
            get_activation_summary(h, 'g.h0.linear')
            h = tf.reshape(h, [-1, self.bottom_height, self.bottom_width, self.feature_maps[0]])

            # residual blocks
            for i, block in enumerate(self.blocks):
                if block != 'attention':
                    h = residual_block(h, output_dim=self.feature_maps[i + 1],
                                       kernel_size=3,
                                       strides=2,
                                       resample=block,
                                       activation_func=tf.nn.relu,
                                       batch_norm_func=batch_norm_func,
                                       use_spectral_norm=use_spectral_norm,
                                       update_spectral_norm=update_spectral_norm,
                                       initializer=self.kernel_initializer,
                                       skip_initializer=self.skip_initializer,
                                       l2_reg=self.l2_reg,
                                       name='g.block{}'.format(i))
                else:
                    h = attention(h,
                                  use_spectral_norm=use_spectral_norm,
                                  update_spectral_norm=update_spectral_norm,
                                  initializer=self.kernel_initializer,
                                  name='g.block{}'.format(i))
                get_activation_summary(h, 'g.block{}'.format(i))

            if batch_norm_func is not None:
                h = batch_norm_func(h, name='g.out.batch_norm')
            h = tf.nn.relu(h, name='g.out.relu')
            h = conv2d(h,
                       output_dim=self.num_img_channels,
                       kernel_size=3,
                       strides=1,
                       use_spectral_norm=use_spectral_norm,
                       update_spectral_norm=update_spectral_norm,
                       l2_reg=self.l2_reg,
                       name='g.out.linear', initializer=None)  # glorot init
            g_out = tf.nn.tanh(h, name='g.out.tanh')
            get_activation_summary(h, 'g.out.linear')
            get_activation_summary(g_out, 'g.out.tanh')
            return g_out


class ResnetDiscriminator(object):
    def __init__(self,
                 feature_maps=128,
                 blocks=('down', 'down', None, None),
                 kernel_initializer=he_initializer,
                 skip_initializer=xavier_initializer,
                 use_batch_norm=True,
                 use_spectral_norm=False,
                 l2_reg=None,
                 name='discriminator', **kwargs):
        self.feature_maps = int2tuple(feature_maps, reps=len(blocks))
        self.blocks = blocks
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.kernel_initializer = kernel_initializer
        self.skip_initializer = skip_initializer
        self.l2_reg = l2_reg
        self.name = name

    def __call__(self, x, training=True, reuse=False, update_spectral_norm=True, **kwargs):
        use_spectral_norm = self.use_spectral_norm
        if not isinstance(self.use_batch_norm, bool):
            batch_norm_func = self.use_batch_norm
        else:
            batch_norm_func = partial(batch_norm,
                                      training=training) if self.use_batch_norm else None
        with tf.variable_scope(self.name, reuse=reuse):
            h = x
            for i, block in enumerate(self.blocks):
                if block != 'attention':
                    h = residual_block(h,
                                       output_dim=self.feature_maps[i],
                                       kernel_size=3,
                                       strides=2,
                                       initializer=self.kernel_initializer,
                                       skip_initializer=self.skip_initializer,
                                       resample=block,
                                       use_spectral_norm=use_spectral_norm,
                                       update_spectral_norm=update_spectral_norm,
                                       batch_norm_func=batch_norm_func,
                                       activation_func=tf.nn.relu,
                                       batch_norm_input=i > 0,
                                       # no need to apply batch normalization to input pixels
                                       activate_input=i > 0,  # don't apply relu to input pixels
                                       l2_reg=self.l2_reg,
                                       name='d.block{}'.format(i))
                else:
                    h = attention(h,
                                  use_spectral_norm=use_spectral_norm,
                                  update_spectral_norm=update_spectral_norm,
                                  initializer=self.kernel_initializer,
                                  name='d.block{}'.format(i))
                get_activation_summary(h, 'd.block{}'.format(i))

            # mean pool layer
            if batch_norm_func is not None:
                h = batch_norm_func(h, name='d.mean_pool.batch_norm')
            h = tf.nn.relu(h, name='d.mean_pool.relu')
            h = tf.reduce_mean(h, axis=[1, 2], name='d.mean_pool.mean')
            get_activation_summary(h, 'd.mean_pool')

            # output layer
            d_out = linear(h, output_dim=1,
                           use_spectral_norm=use_spectral_norm,
                           update_spectral_norm=update_spectral_norm,
                           l2_reg=self.l2_reg,
                           name='d.out')
            get_activation_summary(d_out, 'd.out')
            return d_out
