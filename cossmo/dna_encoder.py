import tensorflow as tf
import numpy as np


_embedding_values = np.zeros([91, 4], np.float32)
_embedding_values[ord('A')] = np.array([1, 0, 0, 0])
_embedding_values[ord('C')] = np.array([0, 1, 0, 0])
_embedding_values[ord('G')] = np.array([0, 0, 1, 0])
_embedding_values[ord('T')] = np.array([0, 0, 0, 1])
_embedding_values[ord('W')] = np.array([.5, 0, 0, .5])
_embedding_values[ord('S')] = np.array([0, .5, .5, 0])
_embedding_values[ord('M')] = np.array([.5, .5, 0, 0])
_embedding_values[ord('K')] = np.array([0, 0, .5, .5])
_embedding_values[ord('R')] = np.array([.5, 0, .5, 0])
_embedding_values[ord('Y')] = np.array([0, .5, 0, .5])
_embedding_values[ord('B')] = np.array([0, 1. / 3, 1. / 3, 1. / 3])
_embedding_values[ord('D')] = np.array([1. / 3, 0, 1. / 3, 1. / 3])
_embedding_values[ord('H')] = np.array([1. / 3, 1. / 3, 0, 1. / 3])
_embedding_values[ord('V')] = np.array([1. / 3, 1. / 3, 1. / 3, 0])
_embedding_values[ord('N')] = np.array([.25, .25, .25, .25])


def dna_encoder(dna_input):
    dna_32 = tf.cast(dna_input, tf.int32)
    embedding_table = tf.get_variable(
        'dna_lookup_table', _embedding_values.shape,
        initializer=tf.constant_initializer(_embedding_values),
        trainable=False)
    encoded_dna = tf.nn.embedding_lookup(embedding_table, dna_32)
    return encoded_dna
