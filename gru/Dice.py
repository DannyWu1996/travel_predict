import tensorflow as tf


def dice(_input, axis=-1, epsilon=1e-9, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable(
            'alpha', _input.get_shape()[-1],
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)

        input_shape = list(_input.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]

        broadcast_shape = [1]*len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    mean = tf.reduce_mean(_input, axis=reduction_axes)
    broadcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_input-broadcast_mean)+epsilon, axis=reduction_axes)
    std = tf.sqrt(std)

    broadcast_std = tf.reshape(std, broadcast_shape)
    input_normed = (_input - broadcast_mean) / (broadcast_std + epsilon)

    input_p = tf.sigmoid(input_normed)

    return alphas*(1.0-input_p)*_input + input_p*_input

def parametric_relu(_input, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable(
            'alphas', _input.get_shape()[-1],
            initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    
        # relu for positive value s
        pos = tf.nn.relu(_input)
        neg = alphas * (_input -abs(_input)) * 0.5
    return pos+neg