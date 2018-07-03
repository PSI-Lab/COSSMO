import tensorflow as tf
import numpy as np
from cossmo.output_networks import BalancedOutputNetwork, RaggedOutputNetwork


class TestCOSSMO(tf.test.TestCase):
    def test_cossmo_predictions(self):
        with self.test_session() as sess:
            num_outputs = 4
            N = 20
            k = 10
            logits_ph = tf.placeholder(tf.float32,
                                       shape=[num_outputs, None, None])

            model = BalancedOutputNetwork(logits_ph, num_outputs, 0., {})
            predictions_t = model.get_psi_predictions()

            feed_dict = {
                logits_ph: np.random.rand(num_outputs, N, k)
            }

            predictions_val = sess.run(predictions_t, feed_dict)

            self.assertTrue(predictions_val.shape, (num_outputs, N, k))
            self.assertTrue(np.allclose(predictions_val.sum(2), 1))

    def test_cossmo_optimizer(self):
        with self.test_session() as sess:
            num_outputs = 4
            N = 20
            k = 10
            H = 15
            X_ph = tf.placeholder(tf.float32,
                                  shape=[num_outputs, N, H])
            W = tf.get_variable('weights', [H, k],
                                initializer=tf.truncated_normal_initializer())
            psi_targets_ph = tf.placeholder(tf.float32,
                                           shape=[num_outputs, None, None])

            logits = tf.reshape(tf.matmul(tf.reshape(X_ph, [-1, H]), W),
                                [num_outputs, -1, k])

            model = BalancedOutputNetwork(logits, num_outputs, 0, {})
            model.get_psi_predictions()
            model.get_cross_entropy_loss(psi_targets_ph)
            model.get_accuracy()
            train_op = model.get_optimizer()

            sess.run(tf.global_variables_initializer())

            feed_dict = {
                X_ph: np.random.rand(num_outputs, N, H),
                psi_targets_ph: np.random.rand(num_outputs, N, k)
            }

            softmax_ce_val, loss_val, accuracy_val = sess.run(
                [model.softmax_cross_entropy, model.loss, model.accuracy],
                feed_dict
            )

            self.assertEqual(softmax_ce_val.shape, (num_outputs, N))
            self.assertIsInstance(loss_val, np.float32)
            self.assertIsInstance(accuracy_val, np.float32)


class TestMaskedCOSSMO(tf.test.TestCase):
    def test_masked_cossmo_predictions(self):
        with self.test_session() as sess:
            num_outputs = 4
            N = 20
            k = 10
            n_alt_ss_val = np.random.randint(0, k, N) + 1
            output_mask = np.array(
                [[1 if j < n_alt_ss_val[i] else 0 for j in range(k)]
                 for i in range(N)]
            ).astype(np.bool)

            logits_ph = tf.placeholder(tf.float32,
                                       shape=[num_outputs, None, None])
            output_mask_ph = tf.placeholder(tf.bool,
                                            shape=[None, None])
            n_alt_ss = tf.placeholder(tf.int32, n_alt_ss_val.shape)

            model = RaggedOutputNetwork(
                logits_ph, num_outputs, n_alt_ss, 0., {})
            predictions_t = model.get_psi_predictions()

            feed_dict = {
                n_alt_ss: n_alt_ss_val,
                logits_ph: np.random.rand(num_outputs, N, k),
                output_mask_ph: output_mask
            }

            predictions_val = sess.run(predictions_t, feed_dict)

            self.assertTrue(predictions_val.shape, (num_outputs, N, k))
            self.assertTrue(np.allclose(predictions_val.sum(2), 1))

    def test_cossmo_optimizer(self):
        with self.test_session() as sess:
            num_outputs = 4
            N = 20
            k = 10
            H = 15
            n_alt_ss_val = np.random.randint(0, k, N) + 1
            output_mask = np.array(
                [[1 if j < n_alt_ss_val[i] else 0 for j in range(k)]
                 for i in range(N)]
            ).astype(np.bool)

            X_ph = tf.placeholder(tf.float32,
                                  shape=[num_outputs, N, H])
            W = tf.get_variable('weights', [H, k],
                                initializer=tf.truncated_normal_initializer())
            psi_targets_ph = tf.placeholder(tf.float32,
                                            shape=[num_outputs, None, None])
            output_mask_ph = tf.placeholder(tf.bool,
                                            shape=[None, None])

            logits = tf.reshape(tf.matmul(tf.reshape(X_ph, [-1, H]), W),
                                [num_outputs, -1, k])

            n_alt_ss = tf.placeholder(tf.int32, n_alt_ss_val.shape)
            model = RaggedOutputNetwork(
                logits, num_outputs, n_alt_ss, 0, {})
            model.get_psi_predictions()
            model.get_cross_entropy_loss(psi_targets_ph)
            model.get_accuracy()
            train_op = model.get_optimizer()

            sess.run(tf.global_variables_initializer())

            feed_dict = {
                n_alt_ss: n_alt_ss_val,
                X_ph: np.random.rand(num_outputs, N, H),
                output_mask_ph: output_mask,
                psi_targets_ph: np.random.rand(num_outputs, N, k)
            }

            softmax_ce_val, loss_val, accuracy_val = sess.run(
                [model.softmax_cross_entropy, model.loss, model.accuracy],
                feed_dict
            )

            self.assertEqual(softmax_ce_val.shape, (num_outputs, N))
            self.assertIsInstance(loss_val, np.float32)
            self.assertIsInstance(accuracy_val, np.float32)
