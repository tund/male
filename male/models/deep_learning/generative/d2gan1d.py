from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import GAN1D
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import linear
from ....backend.tensorflow_backend import minibatch


class D2GAN1D(GAN1D):
    """Dual Discriminator Generative Adversarial Nets for 1D data
    """

    def __init__(self,
                 model_name='D2GAN1D',
                 alpha=1.0,  # coefficient of d1
                 beta=1.0,  # coefficient of d2
                 **kwargs):
        super(D2GAN1D, self).__init__(model_name=model_name, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def _build_model(self, x):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
            self.g = self._create_generator(self.z, self.hidden_size)

        self.x = tf.placeholder(tf.float32, shape=[None, 1])

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('discriminator_1') as scope:
            self.d1x = self._create_discriminator(self.x, self.hidden_size)
            scope.reuse_variables()
            self.d1g = self._create_discriminator(self.g, self.hidden_size)
        with tf.variable_scope('discriminator_2') as scope:
            self.d2x = self._create_discriminator(self.x, self.hidden_size)
            scope.reuse_variables()
            self.d2g = self._create_discriminator(self.g, self.hidden_size)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d1_loss = tf.reduce_mean(-tf.log(self.d1x) + self.d1g)
        self.d2_loss = tf.reduce_mean(self.d2x - tf.log(self.d2g))
        self.d_loss = self.alpha * self.d1_loss + self.beta * self.d2_loss
        self.g_loss = tf.reduce_mean(-self.d1g + tf.log(self.d2g))

        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='discriminator_1') \
                        + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='discriminator_2')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.d_opt = self._create_optimizer(self.d_loss, self.d_params,
                                            self.discriminator_learning_rate)
        self.g_opt = self._create_optimizer(self.g_loss, self.g_params,
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
            d1x, d2x, d1_loss, d2_loss, d_loss, _ = self.tf_session.run(
                [self.d1x, self.d2x, self.d1_loss, self.d2_loss, self.d_loss, self.d_opt],
                feed_dict={self.x: np.reshape(x, [self.batch_size, 1]),
                           self.z: np.reshape(z, [self.batch_size, self.num_z]),
                           })

            # update generator
            z = self.generator.sample([self.batch_size, self.num_z])
            g_loss, _ = self.tf_session.run(
                [self.g_loss, self.g_opt],
                feed_dict={self.z: np.reshape(z, [self.batch_size, self.num_z])})

            batch_logs['d1x'] = d1x.mean()
            batch_logs['d2x'] = d2x.mean()
            batch_logs['d1_loss'] = d1_loss
            batch_logs['d2_loss'] = d2_loss
            batch_logs['d_loss'] = d_loss
            batch_logs['g_loss'] = g_loss
            batch_logs.update(self._on_batch_end(x))
            callbacks.on_batch_end(0, batch_logs)

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

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
        db1 = np.zeros([num_samples, 1])
        db2 = np.zeros([num_samples, 1])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            db1[batch_start:batch_end], db2[batch_start:batch_end] = sess.run(
                [self.d1x, self.d2x],
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

        return db1, db2, pd, pg

    def disp_distribution(self, param, **kwargs):
        db1, db2, pd, pg = self._samples()

        if param == 'avg_distribution':
            self._update_avg_hist(pg)
            pg = self.g_avg_hist['hist']

        db1_x = np.linspace(self.generator.low, self.generator.high, len(db1))
        db2_x = np.linspace(self.generator.low, self.generator.high, len(db2))
        p_x = np.linspace(self.generator.low, self.generator.high, len(pd))

        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.plot(db1_x, db1, label='discriminator D1', linewidth=4)
            ax.plot(db2_x, db2, label='discriminator D2', linewidth=4)
            ax.set_ylim(0, 1.5)
            ax.plot(p_x, pd, label='real data', linewidth=4)
            ax.plot(p_x, pg, label='generated data', linewidth=4)
        else:
            f, ax = plt.subplots(1)
            ax.plot(db1_x, db1, label='discriminator D1', linewidth=4)
            ax.plot(db2_x, db2, label='discriminator D2', linewidth=4)
            ax.set_ylim(0, 1.5)
            plt.plot(p_x, pd, label='real data', linewidth=4)
            plt.plot(p_x, pg, label='generated data', linewidth=4)
            plt.title('1D Dual Discriminator Generative Adversarial Nets', fontsize=28)
            plt.xlabel('Data values', fontsize=28)
            plt.ylabel('Probability density', fontsize=28)
            plt.legend(fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

    def display(self, param, **kwargs):
        if param == 'distribution' or param == 'avg_distribution':
            self.disp_distribution(param, **kwargs)
        else:
            raise NotImplementedError

    def _create_discriminator(self, input, h_dim):
        hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden1'))
        # uncomment to add one more hidden layer, the model will work better.
        # hidden = tf.nn.relu(linear(input, h_dim, 'd_hidden2'))

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if self.minibatch_discriminator:
            hidden = minibatch(hidden, scope='d_minibatch')
        else:
            hidden = tf.nn.relu(linear(hidden, h_dim, scope='d_hidden3'))
        out = tf.nn.softplus(linear(hidden, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate,
                                      beta1=0.5).minimize(loss, var_list=var_list)

    def get_params(self, deep=True):
        out = super(D2GAN1D, self).get_params(deep=deep)
        param_names = D2GAN1D._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
