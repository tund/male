from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf


def get_default_config(per_process_gpu_memory_fraction=None):
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    if per_process_gpu_memory_fraction is not None:
        tf_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    else:
        tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config
