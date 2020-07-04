import tensorflow as tf
import numpy as np

import config


def batch_norm(x, n_out, phase_train=True, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        size = [filter_size, filter_size, in_filters, out_filters]
        init = tf.truncated_normal_initializer(stddev=config.WEIGHT_INIT)
        filters = tf.get_variable('DW', size, tf.float32, init)

        return tf.nn.conv2d(x, filters, [1, strides, strides, 1], 'SAME')


def relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def FC(name, x, output_dim, keep_rate, activation='relu'):
    assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
    with tf.variable_scope(name):
        dim = x.get_shape().as_list()

        # flatten
        dim = np.prod(dim[1:])
        x = tf.reshape(x, [-1, dim])

        # init bias, weight
        W = tf.get_variable('DW', [x.get_shape()[1], output_dim], initializer=tf.truncated_normal_initializer(
                                stddev=config.WEIGHT_INIT))

        b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer())

        x = tf.nn.xw_plus_b(x, W, b)

        # Activation
        if activation == 'relu':
            x = relu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        if activation != 'relu':
            return x
        else:
            return tf.nn.dropout(x, keep_rate)


def max_pool(x, filter_size, strides):
    return tf.nn.max_pool(x, [1, filter_size, filter_size, 1], [1, strides, strides, 1], 'SAME')


def VGG_ConvBlock(name, x, in_filters, out_filters, repeat, strides, phase_train):
    with tf.variable_scope(name):
        for layer in range(repeat):
            scope_name = name + '_' + str(layer)
            x = conv(scope_name, x, 3, in_filters, out_filters, strides)
            x = batch_norm(x, out_filters, phase_train)
            x = relu(x)

            in_filters = out_filters

        x = max_pool(x, 2, 2)
        return x


def get_one_hot_vector(num_classes, class_idx):
    """
        Return tensor of shape (num_classes, )
    """
    result = np.zeros(num_classes)
    result[class_idx] = 1.0

    return result
