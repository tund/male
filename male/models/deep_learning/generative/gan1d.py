from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
config.allow_soft_placement = True

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import Model
from ...distribution import Uniform1D
from ...distribution import Gaussian1D
from ....utils.generic_utils import make_batches


class GAN1D(Model):
    """Generative Adversarial Nets for 1D data
    """

    def __init__(self,
                 model_name='GAN1D',
                 data=None,
                 generator=None,
                 loglik_freq=0,
                 hidden_size=20,
                 minibatch_discriminator=False,
                 generator_learning_rate=0.001,
                 discriminator_learning_rate=0.001,
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(GAN1D, self).__init__(**kwargs)
        self.data = data
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

        tf.set_random_seed(self.random_state)

        self.last_loglik_ = 0.0
        self.g_avg_hist_ = {'count': 0, 'hist': []}

        if self.data is None:
            self.data = Gaussian1D()
        if self.generator is None:
            self.generator = Uniform1D()
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('generator'):
            self.z_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.g_ = self._create_generator(self.z_, self.hidden_size)

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('discriminator') as scope:
            self.x_ = tf.placeholder(tf.float32, shape=[None, 1])
            self.d1_ = self._create_discriminator(self.x_, self.hidden_size)
            scope.reuse_variables()
            self.d2_ = self._create_discriminator(self.g_, self.hidden_size)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss_ = tf.reduce_mean(-tf.log(self.d1_) - tf.log(1 - self.d2_))
        self.g_loss_ = tf.reduce_mean(-tf.log(self.d2_))

        self.d_params_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_params_ = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.d_opt_ = self._create_optimizer(self.d_loss_, self.d_params_,
                                             self.discriminator_learning_rate)
        self.g_opt_ = self._create_optimizer(self.g_loss_, self.g_params_,
                                             self.generator_learning_rate)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            while self.epoch_ < self.num_epochs:
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch_)

                batch_logs = {'batch': 0,
                              'size': self.batch_size}
                callbacks.on_batch_begin(0, batch_logs)

                # update discriminator
                x = self.data.sample(self.batch_size)
                x.sort()
                z = self.generator.stratified_sample(self.batch_size)
                d_loss, _ = sess.run([self.d_loss_, self.d_opt_],
                                     feed_dict={self.x_: np.reshape(x, [self.batch_size, 1]),
                                                self.z_: np.reshape(z, [self.batch_size, 1])})

                # update generator
                z = self.generator.stratified_sample(self.batch_size)
                g_loss, _ = sess.run([self.g_loss_, self.g_opt_],
                                     feed_dict={self.z_: np.reshape(z, [self.batch_size, 1])})

                batch_logs.update(self._on_batch_end(x))
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(0, batch_logs)

                epoch_logs['d_loss'] = d_loss
                epoch_logs['g_loss'] = g_loss
                epoch_logs.update(self._on_epoch_end(sess=sess))
                callbacks.on_epoch_end(self.epoch_, epoch_logs)

                self.epoch_ += 1
                if self.stop_training_:
                    self.epoch_ = self.stop_training_
                    break

    def _on_epoch_end(self, sess=None):
        outs = {}
        if self.loglik_freq != 0:
            if (self.epoch_ + 1) % self.loglik_freq == 0:
                self.last_loglik_ = self.data.logpdf(self.generate(sess=sess))
            outs['loglik'] = self.last_loglik_
        return outs

    def _update_avg_hist(self, hist):
        c = self.g_avg_hist_['count']
        if c == 0:
            self.g_avg_hist_['hist'] = hist
        else:
            self.g_avg_hist_['hist'] = (self.g_avg_hist_['hist'] * c + hist) / (c + 1)
        self.g_avg_hist_['count'] += 1

    def generate(self, num_samples=10000, sess=None):
        close_session = False
        sess = tf.get_default_session() if sess is None else sess
        if sess is None:
            close_session = True
            sess = tf.Session(config=config)
        zs = np.linspace(self.generator.low, self.generator.high, num_samples)
        g = np.zeros([num_samples, 1])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            g[batch_start:batch_end] = sess.run(
                self.g_,
                feed_dict={
                    self.z_: np.reshape(zs[batch_start:batch_end], [batch_end - batch_start, 1])
                }
            )
        if close_session:
            sess.close()
        return g

    def _samples(self, sess, num_samples=10000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        xs = np.linspace(self.generator.low, self.generator.high, num_samples)
        bins = np.linspace(self.generator.low, self.generator.high, num_bins)

        # decision boundary
        db = np.zeros([num_samples, 1])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            db[batch_start:batch_end] = sess.run(
                self.d1_,
                feed_dict={
                    self.x_: np.reshape(xs[batch_start:batch_end], [batch_end - batch_start, 1])
                }
            )

        # data distribution
        d = self.data.sample(num_samples)
        pd, _ = np.histogram(d, bins=bins, density=True)
        pd /= np.sum(pd)

        # generated samples
        g = self.generate(num_samples, sess=sess)
        pg, _ = np.histogram(g, bins=bins, density=True)
        pg /= np.sum(pg)

        return db, pd, pg

    def disp_params(self, param, **kwargs):
        if param == 'distribution' or param == 'avg_distribution':
            sess = tf.get_default_session()

            db, pd, pg = self._samples(sess)

            if param == 'avg_distribution':
                self._update_avg_hist(pg)
                pg = self.g_avg_hist_['hist']

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

    def _create_linear_layer(self, input, output_dim, scope='linear', stddev=0.01):
        norm = tf.random_normal_initializer(stddev=stddev)
        const = tf.constant_initializer(0.0)
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
            b = tf.get_variable('b', [output_dim], initializer=const)
            return tf.matmul(input, w) + b

    def _create_minibatch_layer(self, input, num_kernels=5, kernel_dim=3, scope='minibatch'):
        x = self._create_linear_layer(input, num_kernels * kernel_dim, scope=scope, stddev=0.02)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = (tf.expand_dims(activation, 3)
                 - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0))
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat(1, [input, minibatch_features])

    def _create_generator(self, input, h_dim):
        hidden = tf.nn.softplus(self._create_linear_layer(input, h_dim, 'g_hidden'))
        out = self._create_linear_layer(hidden, 1, 'g_out')
        return out

    def _create_discriminator(self, input, h_dim):
        hidden1 = tf.tanh(self._create_linear_layer(input, h_dim * 2, 'd_hidden1'))
        hidden2 = tf.tanh(self._create_linear_layer(hidden1, h_dim * 2, 'd_hidden2'))

        # without the minibatch layer, the discriminator needs an additional layer
        # to have enough capacity to separate the two distributions correctly
        if self.minibatch_discriminator:
            hidden3 = self._create_minibatch_layer(hidden2, scope='d_minibatch')
        else:
            hidden3 = tf.tanh(self._create_linear_layer(hidden2, h_dim * 2, scope='d_hidden3'))

        out = tf.sigmoid(self._create_linear_layer(hidden3, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate).minimize(loss, var_list=var_list)

    def get_params(self, deep=True):
        out = super(GAN1D, self).get_params(deep=deep)
        out.update({
            'data': self.data,
            'generator': self.generator,
            'loglik_freq': self.loglik_freq,
            'hidden_size': self.hidden_size,
            'minibatch_discriminator': self.minibatch_discriminator,
            'generator_learning_rate': self.generator_learning_rate,
            'discriminator_learning_rate': self.discriminator_learning_rate,
        })
        return out

    def get_all_params(self, deep=True):
        out = super(GAN1D, self).get_all_params(deep=deep)
        out.update({'last_loglik_': self.last_loglik_,
                    'g_avg_hist_': copy.deepcopy(self.g_avg_hist_)})
        return out
