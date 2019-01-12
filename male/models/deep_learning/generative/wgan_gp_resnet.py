from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from .... import TensorFlowModel
from ...distributions import Gaussian1D
from ....utils.generic_utils import make_batches
from ....utils.disp_utils import create_image_grid
from ....metrics import FID, InceptionScore, InceptionMetricList
from .tensorflow_nets import ResnetGenerator, ResnetDiscriminator
from ....backend.tensorflow.ops import he_initializer, adam_optimizer


class WGAN_GP_ResNet(TensorFlowModel):
    """Wasserstein Generative Adversarial Nets with Gradient Penalty using ResNet architecture
    """

    def __init__(self,
                 model_name='WGAN-GP-ResNet',
                 num_z=128,
                 batch_size=64,
                 z_prior=Gaussian1D(mu=0.0, sigma=1.0),
                 d_learning_rate=0.0002,  # SAGAN/BigGAN 0.0004 / 0.0002
                 g_learning_rate=0.00005,  # SAGAN/BigGAN: 0.0001 / 0.00005
                 img_size=(32, 32, 3),  # (height, width, channels)
                 num_gen_feature_maps=128,  # number of feature maps of generator
                 num_dis_feature_maps=128,  # number of feature maps of discriminator
                 g_blocks=('up', 'up', 'up'),
                 d_blocks=('down', 'down', None, None),
                 inception_metrics_freq=int(1e+8),
                 inception_metrics=InceptionMetricList([InceptionScore(), FID(data='cifar10')]),
                 num_inception_samples=100,
                 **kwargs):
        super(WGAN_GP_ResNet, self).__init__(model_name=model_name, **kwargs)
        self.num_z = num_z
        self.batch_size = batch_size
        self.z_prior = z_prior
        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate
        self.img_size = img_size
        self.num_gen_feature_maps = num_gen_feature_maps
        self.num_dis_feature_maps = num_dis_feature_maps
        self.g_blocks = g_blocks
        self.d_blocks = d_blocks
        self.inception_metrics_freq = inception_metrics_freq
        self.inception_metrics = inception_metrics
        self.num_inception_samples = num_inception_samples
        self.log_file = os.path.join(self.log_path, "inception.txt")

    def _init(self):
        super(WGAN_GP_ResNet, self)._init()
        self.z_prior.set_random_engine(self.random_engine)
        if not isinstance(self.inception_metrics, InceptionMetricList):
            if isinstance(self.inception_metrics, list):
                self.inception_metrics = InceptionMetricList(self.inception_metrics)
            else:
                self.inception_metrics = InceptionMetricList([self.inception_metrics])

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1],
                                             self.img_size[2]])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.num_z], name='noise')
        self.training = tf.placeholder(tf.bool, shape=())

        # create generator G and critic
        self.gen = ResnetGenerator(feature_maps=self.num_gen_feature_maps,
                                   img_size=self.img_size,
                                   blocks=self.g_blocks,
                                   use_batch_norm=True,
                                   use_spectral_norm=False,
                                   kernel_initializer=he_initializer,
                                   name='generator')
        self.critic = ResnetDiscriminator(feature_maps=self.num_dis_feature_maps,
                                          blocks=self.d_blocks,
                                          use_batch_norm=False,
                                          use_spectral_norm=False,
                                          kernel_initializer=he_initializer,
                                          name='critic')
        self.g = self.gen(self.z, training=self.training)
        d_x = self.critic(self.x, training=self.training)
        d_g = self.critic(self.g, training=self.training, reuse=True)

        # calculate gradient w.r.t interpolations
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        x_hat = self.x + (self.g - self.x) * epsilon
        d_x_hat = self.critic(x_hat, training=self.training, reuse=True)

        # loss functions
        grad_d_x_hat = tf.gradients(d_x_hat, [x_hat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_hat), axis=[1, 2, 3]))
        self.gradient_penalty = 10.0 * tf.reduce_mean(
            (slopes - 1.0) ** 2)  # the paper set the GP parameter to 10.0

        self.g_loss = -tf.reduce_mean(d_g)
        self.d_loss = tf.reduce_mean(d_g) - tf.reduce_mean(d_x) + self.gradient_penalty

        # create optimizers
        self.d_opt = adam_optimizer(self.d_loss, lr=self.d_learning_rate, beta1=0.0, beta2=0.9,
                                    scope='critic')
        self.g_opt = adam_optimizer(self.g_loss, lr=self.g_learning_rate, beta1=0.0, beta2=0.9,
                                    scope='generator')

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        with open(self.log_file, "w") as file:
            file.write("Start training...\n")
        num_data = x.shape[0] - x.shape[0] % self.batch_size
        callbacks._update_params({'num_samples': num_data})
        batches = make_batches(num_data, self.batch_size)
        self.best_is, self.best_fid = 0, 1000
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            # reshuffle
            indices = np.random.permutation(x.shape[0])
            x_batch, z_batch = None, None

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_size = batch_end - batch_start
                batch_logs = {'batch': batch_idx,
                              'size': batch_size}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                # update critic
                x_batch = x[indices[batch_start:batch_end]]
                z_batch = self.z_prior.sample([self.batch_size, self.num_z]).astype(np.float32)
                d_loss, gradient_penalty, d_opt = self.tf_session.run(
                    [self.d_loss, self.gradient_penalty, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch, self.training: True})

                # update generator G
                z_batch = self.z_prior.sample([self.batch_size, self.num_z]).astype(np.float32)
                g_loss, g_opt = self.tf_session.run(
                    [self.g_loss, self.g_opt],
                    feed_dict={self.z: z_batch, self.training: True})

                # batch_logs.update(self._on_batch_end(x))
                batch_logs['d_loss'] = d_loss
                batch_logs['gradient_penalty'] = gradient_penalty
                batch_logs['g_loss'] = g_loss

                callbacks.on_batch_end(batch_idx, batch_logs)

            self._on_epoch_end(x, epoch_logs,
                               input_data={self.x: x_batch, self.z: z_batch, self.training: True})
            callbacks.on_epoch_end(self.epoch - 1, epoch_logs)

    def _on_epoch_end(self, x, epoch_logs, **kwargs):
        super(WGAN_GP_ResNet, self)._on_epoch_end(**kwargs)
        if self.epoch % self.inception_metrics_freq == 0:
            scores = self.inception_metrics.score(
                self.generate(num_samples=self.num_inception_samples)
            )
            scores = self.inception_metrics.get_score_dict(scores)
            epoch_logs.update(scores)

            self.best_is = max(self.best_is, scores['inception_score'])
            self.best_fid = min(self.best_fid, scores['FID'])
            with open(self.log_file, "a") as file:
                file.write(
                    "Epoch {}:\nIS: Mean {:.2f}  Std {:.2f}  Best {:.2f}\nFID: {:.2f}  Best {:.2f}\n\n".
                        format(self.epoch, scores["inception_score"], scores["inception_score_std"],
                               self.best_is, scores["FID"], self.best_fid))
        else:
            epoch_logs['inception_score'] = 0.0
            epoch_logs['inception_score_std'] = 0.0
            epoch_logs['FID'] = 0.0

    def generate(self, num_samples=100):
        sess = self._get_session()
        num = ((num_samples - 1) // self.batch_size + 1) * self.batch_size
        z = self.z_prior.sample([num, self.num_z]).astype(np.float32)
        x = np.zeros([num, self.img_size[0], self.img_size[1], self.img_size[2]],
                     dtype=np.float32)
        batches = make_batches(num, self.batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            z_batch = z[batch_start:batch_end]
            x[batch_start:batch_end] = sess.run(self.g,
                                                feed_dict={self.z: z_batch, self.training: False})
        if sess != self.tf_session:
            sess.close()
        idx = np.random.permutation(num)[:num_samples]
        return (x[idx] + 1.0) / 2.0

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
        out = super(WGAN_GP_ResNet, self).get_params(deep=deep)
        param_names = WGAN_GP_ResNet._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out

    def _get_params_to_dump(self, deep=True):
        out = dict()
        for key, value in six.iteritems(self.__dict__):
            if ((not type(value).__module__.startswith('tf')) and
                    (not type(value).__module__.startswith('tensorflow')) and
                    (key != 'best_params') and
                    (not isinstance(value, InceptionScore)) and
                    (not isinstance(value, FID)) and
                    (not isinstance(value, InceptionMetricList))):
                out[key] = value
        # param_names = ['tf_graph', 'tf_config', 'tf_merged_summaries']
        # for key in param_names:
        #     if key in self.__dict__:
        #         out[key] = self.__getattribute__(key)
        return out
