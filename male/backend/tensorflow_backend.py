from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from functools import partial


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


batch_norm = partial(tf.contrib.layers.batch_norm,
                     decay=0.9,
                     updates_collections=None,
                     epsilon=1e-5,
                     scale=True)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return tf.nn.bias_add(conv, biases)


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for versions of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def linear(input, output_dim, scope='linear', stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('biases', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def batchnorm(inputs, is_training, decay=0.999, scope='batchnorm'):
    with tf.variable_scope(scope):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, 0.001)


def minibatch(input, num_kernels=5, kernel_dim=3, scope='minibatch'):
    x = linear(input, num_kernels * kernel_dim, scope=scope, stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = (tf.expand_dims(activation, 3)
             - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0))
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [input, minibatch_features])


def rmsprop_optimizer(loss, learning_rate, params):
    opt = tf.train.RMSPropOptimizer(learning_rate)
    grads = opt.compute_gradients(loss, var_list=params)
    train_op = opt.apply_gradients(grads)

    for var in params:
        step = opt.get_slot(var, "momentum")
        update_ratio = tf.abs(step) / (tf.abs(var) + 1e-8)  # update ratio

        tf.summary.histogram(var.op.name + '/values', var)
        tf.summary.histogram(var.op.name + '/update_size', step)
        tf.summary.histogram(var.op.name + '/update_ratio', update_ratio)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return train_op


def adam_optimizer(loss, learning_rate, beta1, params):
    opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)
    grads = opt.compute_gradients(loss, var_list=params)
    train_op = opt.apply_gradients(grads)

    for var in params:
        m = opt.get_slot(var, "m")  # get the first-moment vector
        v = opt.get_slot(var, "v")  # get the second-moment vector

        m_hat = m / (1 - opt._beta1_power)  # bias correction
        v_hat = v / (1 - opt._beta2_power)  # bias correction

        step = learning_rate * m_hat / (v_hat ** 0.5 + opt._epsilon_t)  # update size
        update_ratio = tf.abs(step) / (tf.abs(var) + 1e-8)  # update ratio

        tf.summary.histogram(var.op.name + '/values', var)
        tf.summary.histogram(var.op.name + '/update_size', step)
        tf.summary.histogram(var.op.name + '/update_ratio', update_ratio)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return train_op


def get_active_fraction(x):
    positive = tf.cast(tf.greater(x, 0.), tf.float32)  # 1 if positive, 0 otherwise
    # calculate average times being active across a batch
    batch_positive = tf.reduce_mean(positive, axis=0)
    # define active as being active at least 10% of the batch
    batch_active = tf.greater(batch_positive, 0.1)
    fraction = tf.reduce_mean(tf.cast(batch_active, tf.float32))
    return fraction


def get_activation_summary(x, name):
    tf.summary.histogram(name + '/activations', x)
    tf.summary.scalar(name + '/active_fraction', get_active_fraction(x))
