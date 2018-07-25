import tensorflow as tf
import numpy as np


def W(shape):
    init = tf.truncated_normal(shape, mean=0, stddev=0.01)
    return tf.Variable(init)

def B(shape):
    init = tf.truncated_normal(shape, mean=0, stddev=0.01)
    return tf.Variable(init)

def fc_layer(x, ws, bs):
    w = W(ws)
    b = B(bs)
    net = tf.matmul(x, w) + b
    return tf.nn.relu(net)

def conv2d_layer(x, ws, bs):
    w = W(ws)
    b = B(bs)
    net = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return net

def batch_norm_layer(x):
    fc_mean, fc_var = tf.nn.moments(x, axes=[0])
    scale = tf.Variable(tf.ones(x.shape[-1:]))
    shift = tf.Variable(tf.zeros(x.shape[-1:]))
    x = tf.nn.batch_normalization(x, fc_mean, fc_var, shift, scale, variance_epsilon=0.001)
    return x