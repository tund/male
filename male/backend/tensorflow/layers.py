from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from functools import partial
from .ops import batch_norm, hw_flatten, spectral_norm, he_initializer
from .ops import zero_initializer, dcgan_initializer, xavier_initializer
from .ops import get_activation_summary, get_active_fraction
from ...utils.generic_utils import int2tuple


def linear(input,
           output_dim,
           initializer=he_initializer,
           use_bias=True,
           use_spectral_norm=False,
           update_spectral_norm=True,
           name='linear'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [input.get_shape()[1], output_dim], initializer=initializer)
        if use_spectral_norm:
            lin = tf.matmul(input, spectral_norm(w, update=update_spectral_norm))
        else:
            lin = tf.matmul(input, w)
        if use_bias:
            b = tf.get_variable('biases', [output_dim], initializer=zero_initializer)
            lin = lin + b
        return lin


def conv2d(x,
           output_dim,
           kernel_size=(3, 3),
           strides=(2, 2),
           padding='SAME',
           use_bias=True,
           initializer=dcgan_initializer,
           use_spectral_norm=False,
           update_spectral_norm=True,
           name='conv2d'):
    kernel_size = int2tuple(kernel_size, reps=2)
    strides = int2tuple(strides, reps=2)

    with tf.variable_scope(name):
        w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1],
                                              x.get_shape()[-1], output_dim],
                            initializer=initializer)
        if use_spectral_norm:
            w = spectral_norm(w, update=update_spectral_norm)
        conv = tf.nn.conv2d(input=x, filter=w, padding=padding,
                            strides=[1, strides[0], strides[1], 1])
        if use_bias:
            bias = tf.get_variable('biases', [output_dim], initializer=zero_initializer)
            conv = tf.nn.bias_add(conv, bias)
        return conv


def deconv2d(x,
             output_shape,
             kernel_size=(3, 3),
             strides=(2, 2),
             padding='SAME',
             use_bias=True,
             use_spectral_norm=False,
             update_spectral_norm=True,
             initializer=dcgan_initializer,
             name='deconv2d'):
    kernel_size = int2tuple(kernel_size, reps=2)
    strides = int2tuple(strides, reps=2)

    with tf.variable_scope(name):
        w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1],
                                              output_shape[-1], x.get_shape()[-1]],
                            initializer=initializer)
        if use_spectral_norm:
            w = spectral_norm(w, update=update_spectral_norm)
        deconv = tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape,
                                        strides=[1, strides[0], strides[1], 1],
                                        padding=padding)
        if use_bias:
            bias = tf.get_variable('biases', [output_shape[-1]], initializer=zero_initializer)
            deconv = tf.nn.bias_add(deconv, bias)
        return deconv


def attention_old(x,
                  down_size=8,
                  batch_norm_func=None,
                  activation_func=None,
                  use_spectral_norm=False,
                  update_spectral_norm=True,
                  name='attention'):
    with tf.variable_scope(name):
        z = x
        if batch_norm_func is not None:
            z = batch_norm_func(z, name='input.batchnorm')
        if activation_func is not None:
            z = activation_func(z, name="input.acts")

        f = conv2d(z,
                   output_dim=z.get_shape()[-1] // down_size,
                   kernel_size=1,
                   strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   name='conv.f')
        g = conv2d(z,
                   output_dim=z.get_shape()[-1] // down_size,
                   kernel_size=1,
                   strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   name='conv.g')
        h = conv2d(z,
                   output_dim=z.get_shape()[-1],
                   kernel_size=1,
                   strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   name='conv.h')

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable('gamma', [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


def attention(x,
              initializer=xavier_initializer,
              use_spectral_norm=False,
              update_spectral_norm=True,
              name='attention'):
    #  This is the code from Google Brain's SAGAN
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4

        # theta path
        theta = conv2d(x,
                       output_dim=num_channels // 8,
                       kernel_size=1,
                       strides=1,
                       use_bias=False,
                       initializer=initializer,
                       use_spectral_norm=use_spectral_norm,
                       update_spectral_norm=update_spectral_norm,
                       name='sn_conv_theta')
        theta = tf.reshape(theta, [batch_size, location_num, num_channels // 8])

        # phi path
        phi = conv2d(x,
                     output_dim=num_channels // 8,
                     kernel_size=1,
                     strides=1,
                     use_bias=False,
                     initializer=initializer,
                     use_spectral_norm=use_spectral_norm,
                     update_spectral_norm=update_spectral_norm,
                     name='sn_conv_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(phi, [batch_size, downsampled_num, num_channels // 8])

        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)

        # g path
        g = conv2d(x,
                   output_dim=num_channels // 2,
                   kernel_size=1,
                   strides=1,
                   use_bias=False,
                   initializer=initializer,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   name='sn_conv_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(g, [batch_size, downsampled_num, num_channels // 2])
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_g = conv2d(attn_g,
                        output_dim=num_channels,
                        kernel_size=1,
                        strides=1,
                        use_bias=False,
                        initializer=initializer,
                        use_spectral_norm=use_spectral_norm,
                        update_spectral_norm=update_spectral_norm,
                        name='sn_conv_attn')
        return x + sigma * attn_g


def residual_block(input,
                   output_dim,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   initializer=he_initializer,
                   use_spectral_norm=False,
                   update_spectral_norm=True,
                   batch_norm_func=None,
                   batch_norm_input=True,
                   activation_func=None,
                   activate_input=True,
                   resample=None,
                   name='resblock'):
    kernel_size = int2tuple(kernel_size, reps=2)
    strides = int2tuple(strides, reps=2)

    input_shape = input.get_shape().as_list()
    input_dim = input_shape[-1]
    resize = partial(tf.image.resize_nearest_neighbor,
                     size=(input_shape[1] * strides[0], input_shape[2] * strides[1]))
    mean_pool = partial(tf.nn.avg_pool, ksize=[1, 2, 2, 1],
                        strides=[1, strides[0], strides[1], 1], padding='SAME')

    if resample == 'up':
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, output_dim, kernel_size=1, strides=1,
                          use_spectral_norm=use_spectral_norm,
                          update_spectral_norm=update_spectral_norm,
                          initializer=None, name=name + '.skip.conv')  # Glorot init
        skip = resize(skip, name=name + '.skip.resize')

        # first convolutional layer
        h = input
        if batch_norm_func is not None and batch_norm_input:
            h = batch_norm_func(h, name=name + '.conv1.batch_norm')
        if activation_func is not None and activate_input:
            h = activation_func(h, name=name + '.conv1.acts')
        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv1.lin')

        # second convolutional layer
        if batch_norm_func is not None:
            h = batch_norm_func(h, name=name + '.conv2.batch_norm')
        if activation_func is not None:
            h = activation_func(h, name=name + '.conv2.acts')

        h = resize(h, name=name + '.conv2.resize')
        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv2.lin')

        output = tf.add(h, skip, name=name + '.output')
    elif resample == 'down':
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, output_dim, kernel_size=1, strides=1,
                          use_spectral_norm=use_spectral_norm,
                          update_spectral_norm=update_spectral_norm,
                          initializer=None,
                          name=name + '.skip.conv')  # Glorot init
        skip = mean_pool(skip, name=name + '.skip.mean_pool')

        # first convolutional layer
        h = input
        if batch_norm_func is not None and batch_norm_input:
            h = batch_norm_func(h, name=name + '.conv1.batch_norm')
        if activation_func is not None and activate_input:
            h = activation_func(h, name=name + '.conv1.act')
        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv1.lin')

        # second convolutional layer
        if batch_norm_func is not None:
            h = batch_norm_func(h, name=name + '.conv2.batch_norm')
        if activation_func is not None:
            h = activation_func(h, name=name + '.conv2.acts')

        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv2.lin')
        h = mean_pool(h, name=name + '.conv2.mean_pool')

        output = tf.add(h, skip, name=name + '.output')
    else:  # the same dimension
        # skip connection
        skip = input
        if input_dim != output_dim:
            skip = conv2d(skip, output_dim, kernel_size=1, strides=1,
                          use_spectral_norm=use_spectral_norm,
                          update_spectral_norm=update_spectral_norm,
                          initializer=None, name=name + '.skip.conv')  # Glorot init

        # first convolutional layer
        h = input
        if batch_norm_func is not None and batch_norm_input:
            h = batch_norm_func(h, name=name + '.conv1.batch_norm')
        if activation_func is not None and activate_input:
            h = activation_func(h, name=name + ".conv1.acts")
        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv1.lin')

        # second convolutional layer
        if batch_norm_func is not None:
            h = batch_norm_func(h, name=name + '.conv2.batch_norm')
        if activation_func is not None:
            h = activation_func(h, name=name + '.conv2.acts')
        h = conv2d(h, output_dim, kernel_size=kernel_size, strides=1,
                   use_spectral_norm=use_spectral_norm,
                   update_spectral_norm=update_spectral_norm,
                   initializer=initializer, name=name + '.conv2.lin')

        output = tf.add(h, skip, name=name + '.output')

    return output


def simple_layer(x, output_dim, initializer=he_initializer, training=True, name='dnn', layer=0):
    h = linear(x, output_dim, initializer=initializer, name='{}.layer{}.lin'.format(name, layer))
    h = batch_norm(h, training=training, name='{}.layer{}.batch_norm'.format(name, layer))
    get_activation_summary(h, '{}.layer{}.batch_norm'.format(name, layer))
    h = tf.nn.relu(h, name='{}.layer{}.relu'.format(name, layer))
    get_activation_summary(h, '{}.layer{}.relu'.format(name, layer))
    return h
