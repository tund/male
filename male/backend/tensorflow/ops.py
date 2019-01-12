from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from functools import partial

####################################################################################################
# Weight initializer
####################################################################################################
dcgan_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
he_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN')
xavier_initializer = tf.contrib.layers.xavier_initializer()
zero_initializer = tf.constant_initializer(0.0)
ones_initializer = tf.constant_initializer(1.0)

####################################################################################################
# Normalizations
####################################################################################################
batch_norm = partial(tf.layers.batch_normalization,
                     momentum=0.9,
                     epsilon=1e-5)


def l2_norm(x, eps=1e-12):
    return x / (tf.sqrt(tf.reduce_sum(x ** 2)) + eps)


def spectral_norm_old(w,
                      iterations=1,
                      update=True):
    #  This is the code from Spectral Norm's author
    shape = w.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(w))
    else:
        if len(shape) == 4:
            _w = tf.reshape(w, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        else:
            _w = w

        u = tf.get_variable('singular_vector',
                            [1, shape[0]],
                            initializer=tf.random_normal_initializer,
                            trainable=False)
        _u = u
        for _ in range(iterations):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _w), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_w)), 1)

        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_w, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        if update:
            with tf.control_dependencies([update_u_op]):
                w_bar = w / sigma
        else:
            w_bar = w / sigma

        return w_bar


def spectral_norm(w,
                  num_iters=1,
                  update=True,
                  with_sigma=False):
    #  This is the code from Google Brain's SAGAN
    w_shape = w.shape.as_list()
    w_mat = tf.reshape(w, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = tf.nn.l2_normalize(tf.matmul(u_, w_mat, transpose_b=True), 1)
        u_ = tf.nn.l2_normalize(tf.matmul(v_, w_mat), 1)

    u_ = tf.stop_gradient(u_)
    v_ = tf.stop_gradient(v_)
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def layer_norm(inputs,
               center=True,
               scale=True,
               epsilon=1e-6,
               reuse=False,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               name='layer_norm'):
    inputs_shape = inputs.get_shape().as_list()
    inputs_rank = len(inputs_shape)
    dtype = inputs.dtype.base_dtype

    begin_norm_axis %= inputs_rank
    begin_params_axis %= inputs_rank

    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
        raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                         'must be < rank(inputs) (%d)' %
                         (begin_params_axis, begin_norm_axis, inputs_rank))

    norm_axes = list(range(begin_norm_axis, inputs_rank))
    params_shape = [1] * begin_params_axis + inputs_shape[begin_params_axis:]

    with tf.variable_scope(name, values=[inputs], reuse=reuse):
        if center:
            beta = tf.get_variable('beta',
                                   shape=params_shape,
                                   dtype=dtype,
                                   initializer=zero_initializer,
                                   trainable=trainable)
        if scale:
            gamma = tf.get_variable('gamma',
                                    shape=params_shape,
                                    dtype=dtype,
                                    initializer=ones_initializer,
                                    trainable=trainable)

        # Calculate the moments (instance activations) and outputs
        mean, variance = tf.nn.moments(inputs, norm_axes, keepdims=True)
        inv = tf.rsqrt(variance + epsilon)
        outputs = (inputs - mean) * tf.cast(inv, dtype)
        if scale:
            outputs = outputs * gamma
        if center:
            outputs = outputs + beta
        return outputs


def instance_norm(inputs,
                  center=True,
                  scale=True,
                  epsilon=1e-6,
                  param_initializers=None,
                  reuse=False,
                  trainable=True,
                  data_format='NHWC',
                  name='instance_norm'):
    inputs_shape = inputs.get_shape().as_list()
    inputs_rank = inputs.shape.ndims

    if inputs_rank is None:
        raise ValueError('Inputs %s has undefined rank.' % inputs.name)

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('data_format has to be either NCHW or NHWC.')

    with tf.variable_scope(name, values=[inputs], reuse=reuse):
        if data_format == 'NCHW':
            reduction_axis = 1
            # For NCHW format, rather than relying on implicit broadcasting, we
            # explicitly reshape the params to params_shape_broadcast when computing
            # the moments and the batch normalization.
            params_shape_broadcast = list(
                [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
        else:
            reduction_axis = inputs_rank - 1
            params_shape_broadcast = None

        moments_axes = list(range(inputs_rank))
        del moments_axes[reduction_axis]
        del moments_axes[0]

        params_shape = inputs_shape[reduction_axis:reduction_axis + 1]
        if not params_shape.is_fully_defined():
            raise ValueError(
                'Inputs %s has undefined channels dimension %s.' % (inputs.name, params_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        dtype = inputs.dtype.base_dtype
        if center:
            beta = tf.get_variable('beta',
                                   shape=params_shape,
                                   dtype=dtype,
                                   initializer=zero_initializer,
                                   trainable=trainable)
            if params_shape_broadcast:
                beta = tf.reshape(beta, params_shape_broadcast)
        if scale:
            gamma = tf.get_variable('gamma',
                                    shape=params_shape,
                                    dtype=dtype,
                                    initializer=ones_initializer,
                                    trainable=trainable)
            if params_shape_broadcast:
                gamma = tf.reshape(gamma, params_shape_broadcast)

        # Calculate the moments (instance activations) and outputs
        mean, variance = tf.nn.moments(inputs, moments_axes, keepdims=True)
        inv = tf.rsqrt(variance + epsilon)
        outputs = (inputs - mean) * tf.cast(inv, dtype)
        if scale:
            outputs = outputs * gamma
        if center:
            outputs = outputs + beta
        return outputs


def affine(x, name='affine', reuse=False):
    num_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=reuse):
        beta = tf.get_variable('beta', shape=[1, 1, 1, num_channels], initializer=zero_initializer)
        gamma = tf.get_variable('gamma', shape=[1, 1, 1, num_channels],
                                initializer=ones_initializer)
        outputs = x * gamma + beta
        return outputs


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    return tf.reduce_mean(x, axis=[1, 2])


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


##################################################################################
# Optimizers
##################################################################################

def adam_optimizer(loss, lr=0.0002, beta1=0.0, beta2=0.9,
                   scope="adam", clip=None, summary=True, ignore_list=()):
    def ignore(s):
        for x in ignore_list:
            if s.find(x) >= 0:
                return True

        return False

    params, reg_losses = [], []
    if not isinstance(scope, list) and not isinstance(scope, tuple):
        scope = [scope]

    for sc in scope:
        reg_losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=sc)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
        for var in vars:
            if not ignore(var.op.name):
                params += [var]

    total_loss = tf.add_n([loss] + reg_losses) if len(reg_losses) > 0 else loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2)
        grads = opt.compute_gradients(total_loss, var_list=params)
        if clip is not None:
            grads_clipped = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in grads]
            train_op = opt.apply_gradients(grads_clipped)
        else:
            train_op = opt.apply_gradients(grads)

    for var in params:
        m = opt.get_slot(var, "m")  # get the first-moment vector
        v = opt.get_slot(var, "v")  # get the second-moment vector

        beta1_power, beta2_power = opt._get_beta_accumulators()
        m_hat = m / (1 - beta1_power)  # bias correction
        v_hat = v / (1 - beta2_power)  # bias correction

        step = lr * m_hat / (v_hat ** 0.5 + opt._epsilon_t)  # update size

        if summary:
            tf.summary.histogram(var.op.name + '/values', var)
            tf.summary.histogram(var.op.name + '/update_size', step)

    for grad, var in grads:
        if summary and grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return train_op


def momentum_optimizer(loss, lr=0.0002, momentum=0.9,
                       scope="momentum", clip=None, summary=True, ignore_list=[]):
    def ignore(s):
        for x in ignore_list:
            if s.find(x) >= 0:
                return True

        return False

    params, reg_losses = [], []
    if not isinstance(scope, list) and not isinstance(scope, tuple):
        scope = [scope]

    for sc in scope:
        reg_losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=sc)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
        for var in vars:
            if not ignore(var.op.name):
                params += [var]

    total_loss = tf.add_n([loss] + reg_losses) if len(reg_losses) > 0 else loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.MomentumOptimizer(lr, momentum=momentum)
        grads = opt.compute_gradients(total_loss, var_list=params)
        if clip is not None:
            grads_clipped = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in grads]
            train_op = opt.apply_gradients(grads_clipped)
        else:
            train_op = opt.apply_gradients(grads)

    for grad, var in grads:
        if summary and grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    return train_op


def SGD_optimizer(loss, lr=0.0002, scope="SGD", clip=None, summary=True, ignore_list=[]):
    def ignore(s):
        for x in ignore_list:
            if s.find(x) >= 0:
                return True

        return False

    params, reg_losses = [], []
    if not isinstance(scope, list) and not isinstance(scope, tuple):
        scope = [scope]

    for sc in scope:
        reg_losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=sc)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sc)
        for var in vars:
            if not ignore(var.op.name):
                params += [var]

    total_loss = tf.add_n([loss] + reg_losses) if len(reg_losses) > 0 else loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss, var_list=params)
        if clip is not None:
            grads_clipped = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in grads]
            train_op = opt.apply_gradients(grads_clipped)
        else:
            train_op = opt.apply_gradients(grads)

    for grad, var in grads:
        if summary and grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
            gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))
            tf.summary.scalar(var.op.name + '/gradient_norm', gradient_norm)

    return train_op


####################################################################################################
# Miscs
####################################################################################################

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
