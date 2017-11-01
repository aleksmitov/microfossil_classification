import tensorflow as tf


def conv2d(x, W, strides=(1,1)):
    stride_with = [1, strides[0], strides[1], 1]
    return tf.nn.conv2d(x, W, strides=stride_with, padding='SAME', name='convolution')


def max_pool(x, size=(2,2)):
    ksize = [1, size[0], size[1], 1]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', name='max_pool')


def weight_variable(shape):
    initial_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_val, name='weights')


def bias_variable(shape):
    initial_val = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_val, name='biases')
