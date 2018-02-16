from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel
from ...distribution import Uniform1D
from ....activations import tf_lrelu as lrelu
from ....utils.generic_utils import make_batches
from ....utils.generic_utils import conv_out_size_same
from ....utils.disp_utils import create_image_grid
from ....metrics import InceptionScore
from ....backend.tensorflow_backend import adam_optimizer
from ....backend.tensorflow_backend import get_activation_summary
from ....backend.tensorflow_backend import linear, conv2d, deconv2d


class DCGAN(TensorFlowModel):
    """Deep Convolutional Generative Adversarial Nets
    """

    def __init__(self,
                 model_name='DCGAN',
                 num_z=100,
                 z_prior=Uniform1D(low=-1.0, high=1.0),
                 learning_rate=0.0002,
                 img_size=(32, 32, 3),  # (height, width, channels)
                 num_conv_layers=3,
                 num_gen_feature_maps=128,  # number of feature maps of generator
                 num_dis_feature_maps=128,  # number of feature maps of discriminator
                 inception_score_freq=int(1e+8),
                 num_inception_samples=100,
                 **kwargs):
        super(DCGAN, self).__init__(model_name=model_name, **kwargs)
        self.num_z = num_z
        self.z_prior = z_prior
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.num_conv_layers = num_conv_layers
        self.num_gen_feature_maps = num_gen_feature_maps
        self.num_dis_feature_maps = num_dis_feature_maps
        self.inception_score_freq = inception_score_freq
        self.num_inception_samples = num_inception_samples

    def _init(self):
        super(DCGAN, self)._init()

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
        self.d_opt = adam_optimizer(self.d_loss, self.learning_rate, beta1=0.5, params=d_params)
        self.g_opt = adam_optimizer(self.g_loss, self.learning_rate, beta1=0.5, params=g_params)

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

            if (self.epoch + 1) % self.inception_score_freq == 0 and \
                    "inception_score" in self.metrics:
                epoch_logs['inception_score'] = InceptionScore.inception_score(
                    self.generate(num_samples=self.num_inception_samples))[0]

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end(input_data={self.x: x_batch, self.z: z_batch})

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
                                       scope="g_h0_batchnorm"),
                            name="g_h0_relu")
            h = tf.reshape(h0, [-1, out_size[0][0], out_size[0][1], out_size[0][2]])
            get_activation_summary(h, 'g_h0_relu')

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
                        scope="g_h{}_batchnorm".format(i)),
                    name="g_h{}_relu".format(i))
                get_activation_summary(h, 'g_h{}_relu'.format(i))

            g_out = tf.nn.tanh(
                deconv2d(h, [self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]],
                         stddev=0.02, name="g_out_deconv"),
                name="g_out_tanh")
            get_activation_summary(g_out, 'g_out_tanh')

            return g_out

    def _create_discriminator(self, x, train=True, reuse=False, name="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h = lrelu(conv2d(x, self.num_dis_feature_maps, stddev=0.02, name="d_h0_conv"))
            get_activation_summary(h, "d_h0_lrelu")

            for i in range(1, self.num_conv_layers):
                h = lrelu(batch_norm(conv2d(h, self.num_dis_feature_maps * (2 ** i),
                                            stddev=0.02, name="d_h{}_conv".format(i)),
                                     decay=0.9,
                                     updates_collections=None,
                                     epsilon=1e-5,
                                     scale=True,
                                     is_training=train,
                                     scope="d_h{}_batchnorm".format(i)))
                get_activation_summary(h, "d_h{}_lrelu".format(i))

            dim = h.get_shape()[1:].num_elements()
            d_logits = linear(tf.reshape(h, [-1, dim]), 1,
                              stddev=0.02, scope="d_out_linear")
            d_out = tf.nn.sigmoid(d_logits, name="d_out_sigmoid")
            get_activation_summary(d_out, "d_out_sigmoid")

            return d_out, d_logits

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
        img = create_image_grid(x, img_size=self.img_size, tile_shape=tile_shape,
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

    def generate_images(self, param, **kwargs):
        if param == 'x_samples':
            x = self.generate(num_samples=kwargs['num_samples']) if 'num_samples' in kwargs else \
                self.generate(num_samples=100)
            return create_image_grid(x, img_size=self.img_size, **kwargs)

    def get_params(self, deep=True):
        out = super(DCGAN, self).get_params(deep=deep)
        param_names = DCGAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
