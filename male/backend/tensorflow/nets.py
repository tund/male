from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from functools import partial

from .ops import batch_norm
from .ops import he_initializer
from .layers import linear, conv2d, residual_block, residual_block_bottleneck
from ...utils.generic_utils import int2tuple


class SoftmaxClassifier(object):
    def __init__(self,
                 num_layers=1,
                 hidden_size=256,
                 num_labels=2,
                 activation_func=tf.nn.relu,
                 kernel_initializer=he_initializer,
                 use_batch_norm=True,
                 use_spectral_norm=False,
                 l2_reg=None,
                 name='softmax_classifier', **kwargs):
        self.num_layers = num_layers
        self.hidden_size = int2tuple(hidden_size, reps=num_layers)
        self.num_labels = num_labels
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        self.l2_reg = l2_reg
        self.name = name

    def __call__(self, x, training=True, reuse=False, update_spectral_norm=True, **kwargs):
        batch_norm_func = partial(batch_norm, training=training) if self.use_batch_norm else None
        with tf.variable_scope(self.name, reuse=reuse):
            h = x
            for i in range(self.num_layers):
                h = linear(h,
                           self.hidden_size[i],
                           initializer=self.kernel_initializer,
                           use_spectral_norm=self.use_spectral_norm,
                           update_spectral_norm=update_spectral_norm,
                           l2_reg=self.l2_reg,
                           name='{}.lin{}'.format(self.name, i))
                if batch_norm_func is not None:
                    h = batch_norm_func(h, name='{}.bn{}'.format(self.name, i))
                h = self.activation_func(h, name='{}.atv{}'.format(self.name, i))

            logits = linear(h, self.num_labels, initializer=self.kernel_initializer,
                            use_spectral_norm=self.use_spectral_norm,
                            name='logits')
            return logits


class ResFace2(object):
    '''Just for dirty fixing problems of loading model in a face recognition project.
    '''
    pass
