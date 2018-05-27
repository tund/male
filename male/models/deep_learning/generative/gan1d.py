from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel
from ...distribution import Uniform1D
from ...distribution import Gaussian1D
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import linear


class GAN1D(TensorFlowModel):
    """Generative Adversarial Nets for 1D data
    """

    def __init__(self,
                 model_name='GAN1D',
                 num_z=10,
                 data=None,
                 generator=None,
                 loglik_freq=0,
                 hidden_size=20,
                 minibatch_discriminator=False,
                 generator_learning_rate=0.0001,
                 discriminator_learning_rate=0.0001,
                 **kwargs):
        super(GAN1D, self).__init__(model_name=model_name, **kwargs)
        self.data = data
        self.num_z = num_z
        self.generator = generator
        self.loglik_freq = loglik_freq
        # can use a higher learning rate when not using the minibatch discriminator (MBD) layer
        # e.g., learning rate = 0.005 with MBD, = 0.03 without MBD
        self.minibatch_discriminator = minibatch_discriminator
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.hidden_size = hidden_size

    def _init(self):
        super(GAN1D, self)._init()

        self.last_loglik = 0.0
        self.g_avg_hist = {'count': 0, 'hist': []}

        if self.data is None:
            self.data = Gaussian1D()
        if self.generator is None:
            self.generator = Uniform1D()

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
            self.x = tf.placeholder(tf.float32, shape=[None, 1])
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
                feed_dict={self.x: np.reshape(x, [self.batch_size, 1]),
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

    def _on_batch_end(self, x, y=None):
        outs = super(GAN1D, self)._on_batch_end(x, y)
        if self.loglik_freq != 0:
            if (self.epoch + 1) % self.loglik_freq == 0:
                self.last_loglik = self.data.logpdf(self.generate())
            outs['loglik'] = self.last_loglik
        return outs

    def _update_avg_hist(self, hist):
        c = self.g_avg_hist['count']
        if c == 0:
            self.g_avg_hist['hist'] = hist
        else:
            self.g_avg_hist['hist'] = (self.g_avg_hist['hist'] * c + hist) / (c + 1)
        self.g_avg_hist['count'] += 1

    def generate(self, num_samples=1000):
        sess = self._get_session()
        zs = self.generator.sample([num_samples, self.num_z])
        g = np.zeros([num_samples, 1])
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

    def _samples(self, num_samples=1000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        sess = self._get_session()

        xs = np.linspace(self.generator.low, self.generator.high, num_samples)
        bins = np.linspace(self.generator.low, self.generator.high, num_bins)

        # decision boundary
        db = np.zeros([num_samples, 1])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            db[batch_start:batch_end] = sess.run(
                self.d1,
                feed_dict={
                    self.x: np.reshape(xs[batch_start:batch_end], [batch_end - batch_start, 1])
                }
            )

        # data distribution
        d = self.data.sample(num_samples)
        pd, _ = np.histogram(d, bins=bins, density=True)
        pd /= np.sum(pd)

        # generated samples
        g = self.generate(num_samples)
        pg, _ = np.histogram(g, bins=bins, density=True)
        pg /= np.sum(pg)

        return db, pd, pg

    def display(self, param, **kwargs):
        if param == 'distribution' or param == 'avg_distribution':
            db, pd, pg = self._samples()

            if param == 'avg_distribution':
                self._update_avg_hist(pg)
                pg = self.g_avg_hist['hist']

            db_x = np.linspace(self.generator.low, self.generator.high, len(db))
            p_x = np.linspace(self.generator.low, self.generator.high, len(pd))

            if 'ax' in kwargs:
                ax = kwargs['ax']
                ax.plot(db_x, db, label='decision boundary', linewidth=4)
                ax.set_ylim(0, 1)
                ax.plot(p_x, pd, label='real data', linewidth=4)
                ax.plot(p_x, pg, label='generated data', linewidth=4)
            else:
                f, ax = plt.subplots(1)
                ax.plot(db_x, db, label='decision boundary', linewidth=4)
                ax.set_ylim(0, 1)
                plt.plot(p_x, pd, label='real data', linewidth=4)
                plt.plot(p_x, pg, label='generated data', linewidth=4)
                plt.title('1D Generative Adversarial Network', fontsize=28)
                plt.xlabel('Data values', fontsize=28)
                plt.ylabel('Probability density', fontsize=28)
                plt.legend(fontsize=24)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                plt.show()

    def _create_minibatch_layer(self, input, num_kernels=5, kernel_dim=3, scope='minibatch'):
        x = linear(input, num_kernels * kernel_dim, scope=scope, stddev=0.02)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = (tf.expand_dims(activation, 3)
                 - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0))
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat(1, [input, minibatch_features])

    def _create_generator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'g_hidden1'))
        hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden2'))
        out = linear(hidden, 1, 'g_out')
        return out

    def _create_discriminator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1'))
        # uncomment to add one more hidden layer, the model will work better.
        # hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1'))

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if self.minibatch_discriminator:
            hidden = self._create_minibatch_layer(hidden, scope='d_minibatch')
        else:
            hidden = tf.nn.relu(linear(hidden, h_dim, scope='d_hidden3'))

        out = tf.sigmoid(linear(hidden, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate,
                                      beta1=0.5).minimize(loss, var_list=var_list)

    def get_params(self, deep=True):
        out = super(GAN1D, self).get_params(deep=deep)
        param_names = GAN1D._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
