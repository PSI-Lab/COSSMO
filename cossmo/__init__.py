import tensorflow as tf


def get_unitialized_variables(sess):
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    return uninitialized_vars


def get_dimension(tensor, dim, name='get_dimension'):
    with tf.name_scope(name):
        return tf.unstack(tf.shape(tensor))[dim]

from . import scoring_network
