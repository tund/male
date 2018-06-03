from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import GAN1D
from ...distributions import Gaussian
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import linear


class GAN2D(GAN1D):
    """Generative Adversarial Nets for 2D data
    """

    def __init__(self,
                 model_name='GAN2D',
                 **kwargs):
        super(GAN2D, self).__init__(model_name=model_name, **kwargs)

    def _init(self):
        super(GAN2D, self)._init()

        if self.data is None:
            self.data = Gaussian(mu=(0.0, 0.0), sigma=(1.0, 1.0))

    def _build_model(self, x):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
            self.g = self._create_generator(self.z, self.hidden_size)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('discriminator') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, 2])
            self.d1 = self._create_discriminator(self.x, self.hidden_size)
            scope.reuse_variables()
            self.d2 = self._create_discriminator(self.g, self.hidden_size)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss = tf.reduce_mean(-tf.log(self.d1) - tf.log(1 - self.d2))
        self.g_loss = tf.reduce_mean(-tf.log(self.d2))

        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_opt = self._create_optimizer(self.d_loss, d_params,
                                            self.discriminator_learning_rate)
        self.g_opt = self._create_optimizer(self.g_loss, g_params,
                                            self.generator_learning_rate)

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
            z = self.generator.sample([self.batch_size, self.num_z])
            d_loss, _ = self.tf_session.run(
                [self.d_loss, self.d_opt],
                feed_dict={self.x: np.reshape(x, [self.batch_size, 2]),
                           self.z: np.reshape(z, [self.batch_size, self.num_z])})

            # update generator
            z = self.generator.sample([self.batch_size, self.num_z])
            g_loss, _ = self.tf_session.run(
                [self.g_loss, self.g_opt],
                feed_dict={self.z: np.reshape(z, [self.batch_size, self.num_z])})

            batch_logs['d_loss'] = d_loss
            batch_logs['g_loss'] = g_loss
            batch_logs.update(self._on_batch_end(x))
            callbacks.on_batch_end(0, batch_logs)

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def generate(self, num_samples=1000):
        sess = self._get_session()
        zs = self.generator.sample([num_samples, self.num_z])
        g = np.zeros([num_samples, 2])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            g[batch_start:batch_end] = sess.run(
                self.g,
                feed_dict={
                    self.z: np.reshape(zs[batch_start:batch_end],
                                       [batch_end - batch_start, self.num_z])
                }
            )
        return g

    def disp_scatter(self, **kwargs):
        x = self.data.sample(kwargs['num_samples'])
        g = self.generate(kwargs['num_samples'])
        ax = kwargs['ax']
        ax.scatter(x[:, 0], x[:, 1], s=50, marker='+', color='r', alpha=0.8, label='real data')
        ax.scatter(g[:, 0], g[:, 1], s=50, marker='o', color='b', alpha=0.8, label='generated data')
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    def display(self, param, **kwargs):
        if param == 'scatter':
            self.disp_scatter(**kwargs)
        else:
            raise NotImplementedError

    def _create_generator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden1'))
        hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden2'))
        out = linear(hidden, 2, 'g_out')
        return out

    def _create_discriminator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1'))
        # uncomment the following line to add one more hidden layer, the model will work better.
        # hidden = tf.nn.relu(linear(hidden, h_dim, 'd_hidden2'))
        out = tf.sigmoid(linear(hidden, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate,
                                      beta1=0.5).minimize(loss, var_list=var_list)

    def get_params(self, deep=True):
        out = super(GAN2D, self).get_params(deep=deep)
        param_names = GAN2D._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
