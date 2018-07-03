import unittest
import tensorflow as tf
import numpy as np
from cossmo.dna_encoder import dna_encoder


class TestDnaEncoder(unittest.TestCase):
    def setUp(self):
        self.seq_ = tf.placeholder(tf.uint8)
        self.enc_dna_op = dna_encoder(self.seq_)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def test_dna_encoder(self):
        seq = np.array(['GATTACA', 'ATTACAG', 'AWSMKRY', 'BDHVNNN'], 'c').view(np.uint8)
        enc_seq = self.sess.run(self.enc_dna_op, {self.seq_: seq})
        enc_seq_correct = \
            np.array([[[0., 0., 1., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [1., 0., 0., 0.]],

                      [[1., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [1., 0., 0., 0.],
                       [0., 0., 1., 0.]],

                      [[1., 0., 0., 0.],
                       [0.5, 0., 0., 0.5],
                       [0., 0.5, 0.5, 0.],
                       [0.5, 0.5, 0., 0.],
                       [0., 0., 0.5, 0.5],
                       [0.5, 0., 0.5, 0.],
                       [0., 0.5, 0., 0.5]],

                      [[0., 0.33333334, 0.33333334, 0.33333334],
                       [0.33333334, 0., 0.33333334, 0.33333334],
                       [0.33333334, 0.33333334, 0., 0.33333334],
                       [0.33333334, 0.33333334, 0.33333334, 0.],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25],
                       [0.25, 0.25, 0.25, 0.25]]], dtype=np.float32)
        self.assertTrue(np.allclose(enc_seq, enc_seq_correct), 1e-8)

    def tearDown(self):
        self.sess.close()