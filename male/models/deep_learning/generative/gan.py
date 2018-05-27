from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel
from .... import activations
from ...distribution import Uniform
from ....utils.generic_utils import make_batches
from ....utils.disp_utils import tile_raster_images
from ....backend.tensorflow_backend import linear, batchnorm


class GAN(TensorFlowModel):
    """Generative Adversarial Nets using Multilayer Perceptrons
    """

    def __init__(self,
                 model_name='GAN',
                 num_x=100,
                 discriminator_batchnorm=False,
                 num_discriminator_hiddens=(100,),
                 discriminator_act_funcs=('relu',),
                 discriminator_dropouts=(1.0,),
                 discriminator_learning_rate=0.001,
                 num_z=20,
                 generator_distribution=Uniform(low=(-1.0,) * 20, high=(1.0,) * 20),
                 num_generator_hiddens=(100,),
                 generator_batchnorm=False,
                 generator_act_funcs=('relu',),
                 generator_out_func='sigmoid',
                 generator_learning_rate=0.001,
                 **kwargs):
        super(GAN, self).__init__(model_name=model_name, **kwargs)
        self.num_x = num_x
        self.discriminator_batchnorm = discriminator_batchnorm
        self.num_discriminator_hiddens = num_discriminator_hiddens
        self.discriminator_act_funcs = discriminator_act_funcs
        self.discriminator_dropouts = discriminator_dropouts
        self.discriminator_learning_rate = discriminator_learning_rate
        self.num_z = num_z
        self.generator_batchnorm = generator_batchnorm
        self.generator_distribution = generator_distribution
        self.num_generator_hiddens = num_generator_hiddens
        self.generator_act_funcs = generator_act_funcs
        self.generator_out_func = generator_out_func
        self.generator_learning_rate = generator_learning_rate
        assert self.generator_distribution.dim == self.num_z

    def _build_model(self, x):
        with tf.variable_scope('generator'):
            self.z = tf.placeholder(tf.float32, shape=[None, self.num_z])
            self.g = self._create_generator(self.z,
                                            self.num_generator_hiddens,
                                            self.generator_act_funcs,
                                            self.generator_out_func)

        with tf.variable_scope('discriminator') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, self.num_x])
            self.d1 = self._create_discriminator(self.x,
                                                 self.num_discriminator_hiddens,
                                                 self.discriminator_act_funcs)
            scope.reuse_variables()
            self.d2 = self._create_discriminator(self.g,
                                                 self.num_discriminator_hiddens,
                                                 self.discriminator_act_funcs)

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
        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_size = batch_end - batch_start
                batch_logs = {'batch': batch_idx,
                              'size': batch_size}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                # update discriminator
                x_batch = x[batch_start:batch_end]
                z = self.sample_z(batch_size)
                d_loss, _ = self.tf_session.run(
                    [self.d_loss, self.d_opt],
                    feed_dict={self.x: np.reshape(x_batch, [batch_size, self.num_x]),
                               self.z: np.reshape(z, [batch_size, self.num_z])})

                # update generator
                z = self.sample_z(batch_size)
                g_loss, _ = self.tf_session.run(
                    [self.g_loss, self.g_opt],
                    feed_dict={self.z: np.reshape(z, [batch_size, self.num_z])})

                batch_logs.update(self._on_batch_end(x))
                batch_logs['d_loss'] = d_loss
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def sample_z(self, num_samples):
        return self.generator_distribution.sample(num_samples)

    def generate(self, num_samples=10000):
        sess = self._get_session()
        z = self.sample_z(num_samples)
        x = np.zeros([num_samples, self.num_x])
        batches = make_batches(num_samples, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            x[batch_start:batch_end] = sess.run(
                self.g,
                feed_dict={
                    self.z: np.reshape(z[batch_start:batch_end],
                                       [batch_end - batch_start, self.num_z])
                }
            )
        if sess != self.tf_session:
            sess.close()
        return x

    def _create_generator(self, input, num_hiddens, act_funcs, out_func):
        h = input
        for i in range(len(num_hiddens)):
            linear_layer = linear(h, num_hiddens[i], scope='g_hidden' + str(i + 1), stddev=0.001)
            batchnorm_layer = batchnorm(linear,
                                        is_training=True) \
                if self.generator_batchnorm else linear_layer
            if act_funcs[i] == 'lrelu':
                h = activations.get('tf_' + act_funcs[i])(batchnorm_layer, alpha=0.2)
            else:
                h = activations.get('tf_' + act_funcs[i])(batchnorm_layer)
        out = activations.get('tf_' + out_func)(linear(h, self.num_x, 'g_out'))
        return out

    def _create_discriminator(self, input, num_hiddens, act_funcs):
        h = input
        for i in range(len(num_hiddens)):
            linear_layer = linear(h, num_hiddens[i], scope='d_hidden' + str(i + 1))
            batchnorm_layer = batchnorm(linear,
                                        is_training=True) \
                if self.discriminator_batchnorm else linear_layer
            if act_funcs[i] == 'lrelu':
                h = activations.get('tf_' + act_funcs[i])(batchnorm_layer, alpha=0.2)
            else:
                h = activations.get('tf_' + act_funcs[i])(batchnorm_layer)
            h = tf.nn.dropout(h, self.discriminator_dropouts[i])
        out = tf.sigmoid(linear(h, 1, scope='d_out'))
        return out

    def _create_optimizer(self, loss, var_list, initial_learning_rate):
        return tf.train.AdamOptimizer(initial_learning_rate).minimize(loss, var_list=var_list)

    def disp_generated_data(self, x, disp_dim=None, tile_shape=None,
                            output_pixel_vals=False, **kwargs):
        if disp_dim is None:
            n = int(np.sqrt(x.shape[1]))
            disp_dim = (n, n)
        else:
            assert len(disp_dim) == 2
        if tile_shape is None:
            tile_shape = (x.shape[0], 1)
        img = tile_raster_images(x, img_shape=disp_dim, tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
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

    def get_params(self, deep=True):
        out = super(GAN, self).get_params(deep=deep)
        param_names = GAN._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
