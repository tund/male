from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import tensorflow as tf

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
tf_config.allow_soft_placement = True

from ..linear import GLM
from ... import TensorFlowModel
from ...utils.generic_utils import make_batches


class TensorFlowGLM(TensorFlowModel, GLM):
    """Generalized Linear Model using TensorFlow
    """

    def __init__(self,
                 model_name='TensorFlowGLM',
                 *args, **kwargs):
        kwargs['model_name'] = model_name
        super(TensorFlowGLM, self).__init__(**kwargs)

    def _init(self):
        super(TensorFlowGLM, self)._init()
        self.w_ = None
        self.b_ = None
        self.onehot_encoder_ = None

    def _init_params(self, x):
        # initialize weights
        if self.num_classes_ > 2:
            self.w_ = tf.get_variable("weight", shape=[x.shape[1], self.num_classes_],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
            self.b_ = tf.get_variable("bias", shape=[self.num_classes_],
                                      initializer=tf.constant_initializer(0.0))
        else:
            self.w_ = tf.get_variable("weight", shape=[x.shape[1]],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
            self.b_ = tf.get_variable("bias", shape=[1],
                                      initializer=tf.constant_initializer(0.0))

    def _build_model(self, x):
        self.x_ = tf.placeholder(tf.float32, [None, x.shape[1]])
        if self.num_classes_ > 2:
            self.y_ = tf.placeholder(tf.float32, [None, self.num_classes_])
        else:
            self.y_ = tf.placeholder(tf.float32, [None])
        self.y_link_ = tf.matmul(self.x_, self.w_) + self.b_
        self.y_pred_ = tf.nn.softmax(self.y_link_)
        self.cross_entropy_ = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_pred_), axis=1))
        self.train_func_ = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self.cross_entropy_)
        self.tf_session_.run(tf.global_variables_initializer())

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        batches = make_batches(x.shape[0], self.batch_size)
        while (self.epoch_ < self.num_epochs) and (not self.stop_training_):
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch_)

            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_logs = {'batch': batch_idx,
                              'size': batch_end - batch_start}
                callbacks.on_batch_begin(batch_idx, batch_logs)

                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                self.tf_session_.run(self.train_func_,
                                     feed_dict={self.x_: x_batch, self.y_: y_batch})

                batch_logs.update(self._on_batch_end(x_batch, y_batch))

                callbacks.on_batch_end(batch_idx, batch_logs)

            if do_validation:
                outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                for key, value in outs.items():
                    epoch_logs['val_' + key] = value

            callbacks.on_epoch_end(self.epoch_, epoch_logs)
            self._on_epoch_end()

    def get_loss(self, x, y, *args, **kwargs):
        sess = self._get_session(**kwargs)
        loss = sess.run(self.cross_entropy_, feed_dict={self.x_: x, self.y_: y})
        if sess != self.tf_session_:
            sess.close()
        return loss

    def get_link(self, x, *args, **kwargs):
        sess = self._get_session(**kwargs)
        link = sess.run(self.y_link_, feed_dict={self.x_: x})
        if sess != self.tf_session_:
            sess.close()
        return link

    def get_all_params(self, deep=True):
        out = TensorFlowModel.get_all_params(self, deep=deep)
        out.update(self.get_params(deep=deep))
        out.update({
            'w_': self.w_,
            'b_': self.b_,
            'onehot_encoder_': copy.deepcopy(self.onehot_encoder_),
        })
        return out
