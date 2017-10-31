from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from ..linear import GLM
from ... import TensorFlowModel
from ...utils.generic_utils import make_batches
from ...utils.disp_utils import tile_raster_images

import matplotlib.pyplot as plt

plt.style.use('ggplot')


class TensorFlowGLM(TensorFlowModel, GLM):
    """Generalized Linear Model using TensorFlow
    """

    def __init__(self,
                 model_name='TensorFlowGLM',
                 learning_rate=0.01,
                 **kwargs):
        super(TensorFlowGLM, self).__init__(model_name=model_name, **kwargs)
        self.learning_rate = learning_rate

    def _init_params(self, x):
        # initialize weights
        if self.num_classes > 2:
            self.w = tf.get_variable("weight", shape=[self.data_dim, self.num_classes],
                                     initializer=tf.random_normal_initializer(stddev=0.01))
            self.b = tf.get_variable("bias", shape=[self.num_classes],
                                     initializer=tf.constant_initializer(0.0))
        else:
            self.w = tf.get_variable("weight", shape=[self.data_dim, 1],
                                     initializer=tf.random_normal_initializer(stddev=0.01))
            self.b = tf.get_variable("bias", shape=[1],
                                     initializer=tf.constant_initializer(0.0))

    def _build_model(self, x):
        self.x = tf.placeholder(tf.float32, [None, self.data_dim])
        if self.num_classes > 2:
            self.y = tf.placeholder(tf.float32, [None, self.num_classes])
        else:
            self.y = tf.placeholder(tf.float32, [None])
        self.y_link = tf.matmul(self.x, self.w) + self.b
        if self.loss == 'logit':
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                        logits=tf.squeeze(self.y_link)))
        elif self.loss == 'softmax':
            # self.y_pred = tf.nn.softmax(self.y_link)
            # self.cross_entropy = tf.reduce_mean(
            #     -tf.reduce_sum(self.y * tf.log(self.y_pred), axis=1))
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_link))
        else:  # quadratic (regression)
            self.cross_entropy = 0.5 * tf.reduce_mean(tf.pow(self.y - tf.square(self.y_link), 2))
        self.regularization = 0.0
        if self.l2_penalty > 0:
            self.regularization += 0.5 * self.l2_penalty * tf.reduce_sum(self.w * self.w)
        if self.l1_penalty > 0:
            self.regularization += self.l1_penalty * tf.reduce_sum(tf.abs(self.w))
        self.loss_func = self.cross_entropy + self.regularization
        self.train_func = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.loss_func)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch < self.num_epochs) and (not self.stop_training):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                self.tf_session.run(self.train_func,
                                    feed_dict={self.x: x_batch, self.y: y_batch})

                batch_logs.update(self._on_batch_end(x_batch, y_batch))

                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch, epoch_logs)
            self._on_epoch_end()

    def get_loss(self, x, y, **kwargs):
        sess = self._get_session(**kwargs)
        loss = sess.run(self.cross_entropy, feed_dict={self.x: x, self.y: y})
        if sess != self.tf_session:
            sess.close()
        return loss

    def get_link(self, x, **kwargs):
        sess = self._get_session(**kwargs)
        link = sess.run(self.y_link, feed_dict={self.x: x})
        if sess != self.tf_session:
            sess.close()
        return link

    def disp_weights(self, disp_dim=None, tile_shape=None,
                     output_pixel_vals=False, **kwargs):
        sess = self._get_session(**kwargs)
        w = sess.run(self.w)
        if w.ndim < 2:
            w = w[..., np.newaxis]

        if disp_dim is None:
            n = int(np.sqrt(w.shape[0]))
            disp_dim = (n, n)
        else:
            assert len(disp_dim) == 2
        n = np.prod(disp_dim)

        if tile_shape is None:
            tile_shape = (w.shape[1], 1)
        assert w.shape[1] == np.prod(tile_shape)

        img = tile_raster_images(w.T, img_shape=disp_dim, tile_shape=tile_shape,
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
