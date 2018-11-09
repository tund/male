from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from . import DCGAN
from ....utils.generic_utils import make_batches
from ....backend.tensorflow_backend import conv_cond_concat


class CGAN(DCGAN):
    """Conditional Generative Adversarial Nets (CGAN)
    """

    def __init__(self, model_name="CGAN", **kwargs):
        super(CGAN, self).__init__(model_name=model_name, **kwargs)

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, [self.batch_size,
                                             self.img_size[0], self.img_size[1], self.img_size[2]],
                                name="real_data")
        self.yd = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.num_classes],
                                 name="real_labels")
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.num_z], name='noise')
        self.yg = tf.placeholder(tf.float32, [self.batch_size, self.num_classes],
                                 name="fake_labels")

        # create generator G
        self.g = self._create_generator(tf.concat(values=[self.z, self.yg], axis=1))

        # create sampler to generate samples
        self.sampler = self._create_generator(tf.concat(values=[self.z, self.yg], axis=1),
                                              train=False, reuse=True)

        # create discriminator D
        self.dx, dx_logits = self._create_discriminator(conv_cond_concat(self.x, self.yd))
        self.yg_reshape = tf.reshape(self.yg, [self.batch_size, 1, 1, self.num_classes])
        self.dg, dg_logits = self._create_discriminator(conv_cond_concat(self.g, self.yg_reshape),
                                                        reuse=True)

        # define loss functions
        self.dx_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=dx_logits, labels=tf.ones_like(self.dx)),
            name="dx_loss")
        self.dg_loss = tf.reduce_mean(
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
                y_batch = y[batch_start:batch_end]
                y_onehot = np.zeros([y_batch.shape[0], self.num_classes], dtype=np.float32)
                y_onehot[range(y_batch.shape[0]), y_batch] = 1.0
                z_batch = self.z_prior.sample([batch_size, self.num_z]).astype(np.float32)

                # update discriminator D
                dx_loss, dg_loss, d_loss, _ = self.tf_session.run(
                    [self.dx_loss, self.dg_loss, self.d_loss, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch,
                               self.yd: y_onehot.reshape(-1, 1, 1, self.num_classes),
                               self.yg: y_onehot})

                # update generator G
                g_loss, _ = self.tf_session.run(
                    [self.g_loss, self.g_opt],
                    feed_dict={self.z: z_batch,
                               self.yg: y_onehot})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['dx_loss'] = dx_loss
                batch_logs['dg_loss'] = dg_loss
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            if (self.epoch + 1) % self.inception_score_freq == 0 and \
                            "inception_score" in self.metrics:
                epoch_logs['inception_score'] = self._compute_inception_score(
                    self.generate(num_samples=self.num_inception_samples))

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def generate(self, num_samples=100):
        sess = self._get_session()
        z = self.z_prior.sample([num_samples, self.num_z]).astype(np.float32)
        y = np.arange(self.num_classes).repeat(int(num_samples / self.num_classes))
        y_onehot = np.zeros([y.shape[0], self.num_classes], dtype=np.float32)
        y_onehot[range(y.shape[0]), y] = 1.0
        x = np.zeros([num_samples, self.img_size[0], self.img_size[1], self.img_size[2]],
                     dtype=np.float32)
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_size = batch_end - batch_start
            z_batch = z[batch_start:batch_end]
            y_batch = y_onehot[batch_start:batch_end]
            if batch_size < self.batch_size:
                z_batch = np.vstack(
                    [z_batch, np.zeros([self.batch_size - batch_size,
                                        self.num_z])])
                y_addition = np.zeros([self.batch_size - batch_size, self.num_classes])
                y_addition[:, 0] = 1.0
                y_batch = np.vstack([y_batch, y_addition])
            x[batch_start:batch_end] = sess.run(
                self.sampler,
                feed_dict={self.z: z_batch, self.yg: y_batch})[:batch_size]
        if sess != self.tf_session:
            sess.close()
        return (x + 1.0) / 2.0


class CGANv1(CGAN):
    """Conditional Generative Adversarial Nets (CGAN) -- version 1.0
    """

    def __init__(self, model_name="CGANv1", **kwargs):
        super(CGANv1, self).__init__(model_name=model_name, **kwargs)

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
                y_batch = y[batch_start:batch_end]
                y_onehot = np.zeros([y_batch.shape[0], self.num_classes], dtype=np.float32)
                y_onehot[range(y_batch.shape[0]), y_batch] = 1.0
                z_batch = self.z_prior.sample([batch_size, self.num_z]).astype(np.float32)

                # update discriminator D
                dx_loss, dg_loss, d_loss, _ = self.tf_session.run(
                    [self.dx_loss, self.dg_loss, self.d_loss, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch,
                               self.yd: y_onehot.reshape(-1, 1, 1, self.num_classes),
                               self.yg: y_onehot})

                # update generator G
                y_onehot = np.zeros([y_batch.shape[0], self.num_classes], dtype=np.float32)
                y_onehot[range(y_batch.shape[0]),
                         np.random.randint(0, self.num_classes, size=y_batch.shape[0])] = 1.0
                g_loss, _ = self.tf_session.run(
                    [self.g_loss, self.g_opt],
                    feed_dict={self.z: z_batch,
                               self.yg: y_onehot})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['dx_loss'] = dx_loss
                batch_logs['dg_loss'] = dg_loss
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            if (self.epoch + 1) % self.inception_score_freq == 0 and \
                            "inception_score" in self.metrics:
                epoch_logs['inception_score'] = self._compute_inception_score(
                    self.generate(num_samples=self.num_inception_samples))

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()
