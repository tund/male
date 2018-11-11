from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config
