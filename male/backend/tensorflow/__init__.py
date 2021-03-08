from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from packaging import version

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


def set_memory_growth():
    if version.parse(tf.__version__) >= version.parse('2.0'):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    else:
        raise NotImplementedError
