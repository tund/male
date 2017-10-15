from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel
from ...distribution import Uniform1D
from ...distribution import Gaussian
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import linear


class MGGAN2D(TensorFlowModel):
    """Mix-Generator Generative Adversarial Nets (MGGAN) for 2D data
    """

    def __init__(self,
                 model_name='MGGAN2D',
                 beta=3,
                 num_z=10,
                 num_gens=2,
                 d_batch_size=64,
                 g_batch_size=32,
                 data=None,
                 generator=None,
                 loglik_freq=0,
                 d_hidden_size=20,
                 g_hidden_size=20,
                 generator_learning_rate=0.0001,
                 discriminator_learning_rate=0.0001,
                 **kwargs):
        super(MGGAN2D, self).__init__(model_name=model_name, **kwargs)
        self.beta = beta
        self.data = data
        self.num_z = num_z
        self.num_gens = num_gens
        self.d_batch_size = d_batch_size
        self.g_batch_size = g_batch_size
        self.generator = generator
        self.loglik_freq = loglik_freq
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.d_hidden_size = d_hidden_size
        self.g_hidden_size = g_hidden_size

    def _init(self):
        super(MGGAN2D, self)._init()

        self.last_loglik = 0.0
        self.g_avg_hist = {'count': 0, 'hist': []}

        if self.data is None:
            self.data = Gaussian(mu=(0.0, 0.0), sigma=(1.0, 1.0))
        if self.generator is None:
            self.generator = Uniform1D()

    def _build_model(self, x):
        arr1 = np.array([i // self.g_batch_size for i in range(self.g_batch_size * self.num_gens)])
        arr2 = np.array([1 if i // (self.g_batch_size * self.num_gens) == i % self.num_gens
                         else 0
                         for i in range(self.g_batch_size * self.num_gens * self.num_gens)])
        arr2 = arr2.reshape(self.g_batch_size * self.num_gens, self.num_gens)

        d_mul_labels = tf.constant(arr1, dtype=tf.int32)
        g_mul_labels = tf.constant(arr2, dtype=tf.float32)

        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
            self.g = self._create_generator(self.z, self.g_hidden_size)

        with tf.variable_scope('discriminator') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, 2])
            d_bin_x_logits, d_mul_x_logits = self._create_discriminator(self.x, self.d_hidden_size)
            scope.reuse_variables()
            d_bin_g_logits, d_mul_g_logits = self._create_discriminator(self.g, self.d_hidden_size)

        # Define the loss function
        self.d_bin_x_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_x_logits, labels=tf.ones_like(d_bin_x_logits)),
            name='d_bin_x_loss')
        self.d_bin_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_g_logits, labels=tf.zeros_like(d_bin_g_logits)),
            name='d_bin_g_loss')
        self.d_bin_loss = tf.add(self.d_bin_x_loss, self.d_bin_g_loss, name='d_bin_loss')
        self.d_mul_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_mul_g_logits, labels=d_mul_labels),
            name="d_mul_loss")
        self.d_loss = tf.add(self.d_bin_loss, self.d_mul_loss, name="d_loss")
        # self.d_loss = tf.identity(self.d_bin_loss, name='d_loss')

        self.g_bin_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_g_logits, labels=tf.ones_like(d_bin_g_logits)),
            name="g_bin_loss")
        self.g_mul_loss = -self.beta * tf.reduce_mean(
            tf.nn.log_softmax(d_mul_g_logits) * g_mul_labels,
            name='g_mul_loss')
        self.g_loss = tf.add(self.g_bin_loss, self.g_mul_loss, name="g_loss")
        # self.g_loss = tf.identity(self.g_bin_loss, name='g_loss')

        self.d_opt = self._create_optimizer(self.d_loss, scope='discriminator',
                                            lr=self.discriminator_learning_rate)
        self.g_opt = self._create_optimizer(self.g_loss, scope='generator',
                                            lr=self.generator_learning_rate)
        self.merged_summaries = tf.summary.merge_all()

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            batch_logs = {'batch': 0,
                          'size': self.d_batch_size}
            callbacks.on_batch_begin(0, batch_logs)

            # update discriminator
            x = self.data.sample(self.d_batch_size)
            z = self.generator.sample([self.g_batch_size, self.num_z])
            _summary, d_bin_loss, d_mul_loss, d_loss, _ = self.tf_session.run(
                [self.tf_merged_summaries, self.d_bin_loss, self.d_mul_loss, self.d_loss, self.d_opt],
                feed_dict={self.x: np.reshape(x, [self.d_batch_size, 2]),
                           self.z: np.reshape(z, [self.g_batch_size, self.num_z])})

            # update generator
            z = self.generator.sample([self.g_batch_size, self.num_z])
            g_bin_loss, g_mul_loss, g_loss, _ = self.tf_session.run(
                [self.g_bin_loss, self.g_mul_loss, self.g_loss, self.g_opt],
                feed_dict={self.z: np.reshape(z, [self.g_batch_size, self.num_z])})

            if (self.epoch + 1) % self.summary_freq == 0:
                _summary = self.tf_session.run(
                    self.tf_merged_summaries,
                    feed_dict={self.x: np.reshape(x, [self.d_batch_size, 2]),
                               self.z: np.reshape(z, [self.g_batch_size, self.num_z])})
                self.tf_summary_writer.add_summary(_summary, self.epoch + 1)

            batch_logs['d_loss'] = d_loss
            batch_logs['d_bin_loss'] = d_bin_loss
            batch_logs['d_mul_loss'] = d_mul_loss
            batch_logs['g_loss'] = g_loss
            batch_logs['g_bin_loss'] = g_bin_loss
            batch_logs['g_mul_loss'] = g_mul_loss
            batch_logs.update(self._on_batch_end(x))
            callbacks.on_batch_end(0, batch_logs)

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def _on_batch_end(self, x, y=None):
        outs = super(MGGAN2D, self)._on_batch_end(x, y)
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
        batch_size = self.g_batch_size * self.num_gens
        num = ((num_samples - 1) // batch_size + 1) * batch_size
        g = np.zeros([num, 2])
        batches = make_batches(num, batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            z_batch = self.generator.sample([self.g_batch_size, self.num_z])
            g[batch_start:batch_end] = sess.run(
                self.g,
                feed_dict={
                    self.z: np.reshape(z_batch,
                                       [self.g_batch_size, self.num_z])
                }
            )
        idx = np.random.permutation(num)[:num_samples]
        return g[idx]

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
        if param == 'scatter':
            self.disp_scatter(**kwargs)
        else:
            raise NotImplementedError

    def _create_generator(self, input, h_dim):
        hidden = []
        for i in range(self.num_gens):
            hidden.append(tf.nn.relu(linear(input, h_dim, 'g_hidden1_{}'.format(i))))
            self._activation_summary(hidden[-1], 'g_hidden1_{}'.format(i))
        hidden = tf.concat(hidden, axis=0, name='g_hidden1')
        hidden = tf.nn.relu(linear(hidden, h_dim, 'g_hidden2'))
        self._activation_summary(hidden, 'g_hidden2')
        out = linear(hidden, 2, 'g_out')
        # h_split = tf.split(hidden, self.num_gens, axis=0)
        # out = []
        # for i, var in enumerate(h_split):
        #    out.append(linear(var, 2, 'g_out{}'.format(i)))
        #    self._activation_summary(out[-1], 'g_out_{}'.format(i))
        # out = tf.concat(out, axis=0, name='g_out')
        return out

    def _create_discriminator(self, input, h_dim):
        hidden = lrelu(linear(input, h_dim, 'd_hidden1'))
        self._activation_summary(hidden, 'd_hidden1')
        # uncomment to add one more hidden layer, the model will work better.
        # hidden = lrelu(linear(hidden, h_dim, 'd_hidden2'))
        # self._activation_summary(hidden, 'd_hidden2')
        d_bin_logits = linear(hidden, 1, scope='d_bin_logits')
        self._activation_summary(d_bin_logits, 'd_bin_logits')
        d_mul_logits = linear(hidden, self.num_gens, scope='d_mul_logits')
        self._activation_summary(d_mul_logits, 'd_mul_logits')
        return d_bin_logits, d_mul_logits

    def _create_optimizer(self, loss, scope, lr):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5)
        grads = opt.compute_gradients(loss, var_list=params)
        train_op = opt.apply_gradients(grads)

        for var in params:
            tf.summary.histogram(var.op.name + '/values', var)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        return train_op

    def _active_fraction(self, x):
        positive = tf.cast(tf.greater(x, 0.), tf.float32)  # 1 if positive, 0 otherwise
        batch_positive = tf.reduce_mean(positive, axis=0)  # calculate average times being active across a batch
        batch_active = tf.greater(batch_positive, 0.1)  # define active as being active at least 10% of the batch
        fraction = tf.reduce_mean(tf.cast(batch_active, tf.float32))
        return fraction

    def _activation_summary(self, x, name):
        tf.summary.histogram(name + '/activations', x)
        tf.summary.scalar(name + '/active_fraction', self._active_fraction(x))

    def get_params(self, deep=True):
        out = super(MGGAN2D, self).get_params(deep=deep)
        param_names = MGGAN2D._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
