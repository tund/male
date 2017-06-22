from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import DCGAN
from ...distribution import Uniform1D


class WeightedGAN(DCGAN):
    """Weighted Generative Adversarial Nets
    """

    def __init__(self,
                 model_name='WeightedGAN',
                 alpha=0.5,
                 **kwargs):
        super(WeightedGAN, self).__init__(model_name=model_name, **kwargs)
        self.alpha = alpha

    def _init(self):
        super(WeightedGAN, self)._init()

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
        self.dx, dx_logits = self._create_discriminator(self.x)
        self.dg, dg_logits = self._create_discriminator(self.g, reuse=True)

        # define loss functions
        self.dx_loss = self.alpha * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dx_logits, labels=tf.ones_like(self.dx)),
            name="dx_loss")
        self.dg_loss = (1 - self.alpha) * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dg_logits, labels=tf.zeros_like(self.dg)),
            name="dg_loss")
        self.d_loss = tf.add(self.dx_loss, self.dg_loss, name="d_loss")
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dg_logits, labels=tf.ones_like(self.dg)),
            name="g_loss")

        # create optimizers
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.d_loss, var_list=d_params)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5) \
            .minimize(self.g_loss, var_list=g_params)

    def get_params(self, deep=True):
        out = super(WeightedGAN, self).get_params(deep=deep)
        param_names = WeightedGAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
