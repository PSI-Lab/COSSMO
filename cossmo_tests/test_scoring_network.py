import tensorflow as tf
import numpy as np
from cossmo.scoring_network import ScoringNetwork, MIN_FLOAT
from cossmo.output_networks import BalancedOutputNetwork, RaggedOutputNetwork


class TestConvNetScoringNetwork(tf.test.TestCase):
    def test_scoring_network_one_output(self):
        with self.test_session() as sess:
            k = 3
            exonic_seq_length = 7
            intronic_seq_length = 7
            seq_length = exonic_seq_length + intronic_seq_length
            conv_params = [(3, 2, 1, 3, 1)]
            hidden_units = [10]
            n_outputs = 1
            N = 2

            with tf.variable_scope('score_one_output'):
                rna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='rna_input')
                const_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, seq_length],
                    name='const_dna_input')
                alt_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='alt_dna_input')

                network_inputs = {
                    'rna_seq': rna_input_ph,
                    'const_dna_seq': const_dna_input_ph,
                    'alt_dna_seq': alt_dna_input_ph
                }

                network_parameters = {
                    'exonic_seq_length': exonic_seq_length,
                    'intronic_seq_length': intronic_seq_length,
                    'conv_params': conv_params,
                    'hidden_units': hidden_units,
                    'n_outputs': n_outputs
                }

                scoring_network = \
                    ScoringNetwork(
                        network_inputs,
                        network_parameters,
                        features=[]
                    )

            sess.run(tf.global_variables_initializer())

            rna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_dna_input = np.array(N * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape(N, seq_length)
            alt_dna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))

            logit = sess.run(
                fetches=scoring_network.outputs['logit'],
                feed_dict={
                    rna_input_ph: rna_input,
                    const_dna_input_ph: const_dna_input,
                    alt_dna_input_ph: alt_dna_input
                }
            )

            self.assertEqual(logit.shape, (n_outputs, N, k))

    def test_scoring_loss(self):
        with self.test_session() as sess:
            k = 3
            exonic_seq_length = 7
            intronic_seq_length = 7
            seq_length = exonic_seq_length + intronic_seq_length
            conv_params = [(3, 2, 1, 3, 1)]
            hidden_units = [10]
            n_outputs = 5
            N = 2

            with tf.variable_scope('scoring_loss'):
                rna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='rna_input')
                const_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, seq_length],
                    name='const_dna_input')
                alt_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='alt_dna_input')
                psi_targets_ph = tf.placeholder(
                    tf.float32, shape=[n_outputs, None, None]
                )

                network_inputs = {
                    'rna_seq': rna_input_ph,
                    'const_dna_seq': const_dna_input_ph,
                    'alt_dna_seq': alt_dna_input_ph
                }

                network_parameters = {
                    'exonic_seq_length': exonic_seq_length,
                    'intronic_seq_length': intronic_seq_length,
                    'conv_params': conv_params,
                    'hidden_units': hidden_units,
                    'n_outputs': n_outputs
                }

                scoring_network = \
                    ScoringNetwork(
                        network_inputs,
                        network_parameters,
                        features=[]
                    )

                cossmo = BalancedOutputNetwork(
                    scoring_network.outputs['logit'], n_outputs, 0., {})
                cossmo.get_psi_predictions()
                cossmo.get_cross_entropy_loss(psi_targets_ph)
                cossmo.get_optimizer()

            sess.run(tf.global_variables_initializer())

            rna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_dna_input = np.array(N * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape(N, seq_length)
            alt_dna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            psi_targets = np.random.random((n_outputs, N, k))

            loss, cross_entropy = \
                sess.run(fetches=[cossmo.loss,
                                  cossmo.softmax_cross_entropy],
                              feed_dict={
                                  rna_input_ph: rna_input,
                                  const_dna_input_ph: const_dna_input,
                                  alt_dna_input_ph: alt_dna_input,
                                  psi_targets_ph: psi_targets
                              }
                              )
            self.assertIsInstance(loss, np.float32)
            self.assertEqual(cross_entropy.shape, (n_outputs, N))

    def test_var_output(self):
        with self.test_session():
            k = 5
            exonic_seq_length = 7
            intronic_seq_length = 7
            seq_length = exonic_seq_length + intronic_seq_length
            conv_params = [(3, 2, 1, 3, 1)]
            hidden_units = [10]
            n_outputs = 5
            N = 20
            n_alt_ss = np.random.randint(2, k+1, N).astype(np.int32)

            with tf.variable_scope('scoring_loss'):
                n_alt_ss_ph = tf.placeholder(tf.int32, shape=[None],
                                             name='n_alt_ss')
                rna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='rna_input')
                const_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, seq_length],
                    name='const_dna_input')
                alt_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[None, k, seq_length],
                    name='alt_dna_input')
                psi_targets_ph = tf.placeholder(
                    tf.float32, shape=[n_outputs, None, None]
                )
                output_mask_ph = tf.placeholder(
                    tf.bool, shape=[None, None]
                )

                network_inputs = {
                    'rna_seq': rna_input_ph,
                    'const_dna_seq': const_dna_input_ph,
                    'alt_dna_seq': alt_dna_input_ph
                }

                network_parameters = {
                    'exonic_seq_length': exonic_seq_length,
                    'intronic_seq_length': intronic_seq_length,
                    'conv_params': conv_params,
                    'hidden_units': hidden_units,
                    'n_outputs': n_outputs
                }

                scoring_network = \
                    ScoringNetwork(
                        network_inputs,
                        network_parameters,
                        features=[]
                    )

                cossmo = RaggedOutputNetwork(
                    scoring_network.outputs['logit'],
                    n_outputs, n_alt_ss_ph,
                    0., {}
                )
                cossmo.get_psi_predictions()
                cossmo.get_cross_entropy_loss(psi_targets_ph)
                cossmo.get_optimizer()

            tf.global_variables_initializer().run()

            rna_input = np.array(N * k * ['GATTACAGATTACA'], 'c') \
                .view(np.uint8).reshape((N, k, seq_length))
            const_dna_input = np.array(N * ['GATTACAGATTACA'], 'c') \
                .view(np.uint8).reshape(N, seq_length)
            alt_dna_input = np.array(N * k * ['GATTACAGATTACA'], 'c') \
                .view(np.uint8).reshape((N, k, seq_length))
            psi_targets = np.zeros((n_outputs, N, k), np.float32)
            output_mask = np.ones((N, k)).astype(np.bool)
            for j in range(N):
                output_mask[j,n_alt_ss[j]:] = False

            for i in range(n_outputs):
                for j in range(N):
                    psi_targets[i, j, :n_alt_ss[j]] = \
                        np.random.random((n_alt_ss[j]))
            psi_targets /= psi_targets.sum(2, keepdims=True)

            feed_dict = {
                n_alt_ss_ph: n_alt_ss,
                rna_input_ph: rna_input,
                const_dna_input_ph: const_dna_input,
                alt_dna_input_ph: alt_dna_input,
                psi_targets_ph: psi_targets,
                output_mask_ph: output_mask
            }

            loss = cossmo.loss.eval(feed_dict)
            cross_entropy = cossmo.softmax_cross_entropy.\
                eval(feed_dict)
            masked_logit = cossmo.masked_logit.eval(feed_dict)

            self.assertTrue(np.isfinite(loss))
            self.assertTrue(np.all(np.isfinite(cross_entropy)))
            for i in range(n_outputs):
                for j in range(N):
                    masked_entries = masked_logit[i, j, n_alt_ss[j]:]
                    self.assertAllCloseAccordingToType(
                        masked_entries,
                        MIN_FLOAT * np.ones_like(masked_entries)
                    )

    def test_intron_length(self):
        with self.test_session() as sess:
            k = 3
            exonic_seq_length = 7
            intronic_seq_length = 7
            seq_length = exonic_seq_length + intronic_seq_length
            conv_params = [(3, 2, 1, 3, 1)]
            hidden_units = [10]
            n_outputs = 1
            N = 2

            with tf.variable_scope('score_one_output'):
                rna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, k, seq_length],
                    name='rna_input')
                const_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, seq_length],
                    name='const_dna_input')
                alt_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, k, seq_length],
                    name='alt_dna_input')
                const_site_pos_ph = tf.placeholder(
                    tf.int32, shape=[N],
                    name='const_site_position'
                )
                alt_site_pos_ph = tf.placeholder(
                    tf.int32, shape=[N, k],
                    name='alt_site_positions'
                )

                network_inputs = {
                    'rna_seq': rna_input_ph,
                    'const_dna_seq': const_dna_input_ph,
                    'alt_dna_seq': alt_dna_input_ph,
                    'const_site_position': const_site_pos_ph,
                    'alt_ss_position': alt_site_pos_ph
                }

                network_parameters = {
                    'exonic_seq_length': exonic_seq_length,
                    'intronic_seq_length': intronic_seq_length,
                    'conv_params': conv_params,
                    'hidden_units': hidden_units,
                    'n_outputs': n_outputs
                }

                scoring_network = \
                    ScoringNetwork(
                        network_inputs,
                        network_parameters
                    )

            sess.run(tf.global_variables_initializer())

            rna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_dna_input = np.array(N * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape(N, seq_length)
            alt_dna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_site_pos_input = np.array(N * [0], np.int32)
            alt_site_pos_input = np.array(N * k * [1000], np.int32)\
                .reshape(N, k)

            intron_length, intron_length_norm = sess.run(
                fetches=[scoring_network.nodes['intron_length'],
                         scoring_network.nodes['intron_length_norm']],
                feed_dict={
                    rna_input_ph: rna_input,
                    const_dna_input_ph: const_dna_input,
                    alt_dna_input_ph: alt_dna_input,
                    const_site_pos_ph: const_site_pos_input,
                    alt_site_pos_ph: alt_site_pos_input
                }
            )

            self.assertEqual(intron_length[0,0], 1000)
            self.assertAlmostEqual(intron_length_norm[0,0], -0.33795869)

    def test_logistic_regression(self):
        with self.test_session() as sess:
            k = 3
            exonic_seq_length = 7
            intronic_seq_length = 7
            seq_length = exonic_seq_length + intronic_seq_length
            conv_params = []
            hidden_units = []
            n_outputs = 1
            N = 2

            with tf.variable_scope('score_one_output'):
                rna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, k, seq_length],
                    name='rna_input')
                const_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, seq_length],
                    name='const_dna_input')
                alt_dna_input_ph = tf.placeholder(
                    tf.uint8, shape=[N, k, seq_length],
                    name='alt_dna_input')
                const_site_pos_ph = tf.placeholder(
                    tf.int32, shape=[N],
                    name='const_site_position'
                )
                alt_site_pos_ph = tf.placeholder(
                    tf.int32, shape=[N, k],
                    name='alt_site_positions'
                )

                network_inputs = {
                    'rna_seq': rna_input_ph,
                    'const_dna_seq': const_dna_input_ph,
                    'alt_dna_seq': alt_dna_input_ph,
                    'const_site_position': const_site_pos_ph,
                    'alt_ss_position': alt_site_pos_ph
                }

                network_parameters = {
                    'exonic_seq_length': exonic_seq_length,
                    'intronic_seq_length': intronic_seq_length,
                    'conv_params': conv_params,
                    'hidden_units': hidden_units,
                    'n_outputs': n_outputs
                }

                scoring_network = \
                    ScoringNetwork(
                        network_inputs,
                        network_parameters
                    )

            sess.run(tf.global_variables_initializer())

            rna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_dna_input = np.array(N * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape(N, seq_length)
            alt_dna_input = np.array(N * k * ['GATTACAGATTACA'], 'c')\
                .view(np.uint8).reshape((N, k, seq_length))
            const_site_pos_input = np.array(N * [0], np.int32)
            alt_site_pos_input = np.array(N * k * [1000], np.int32)\
                .reshape(N, k)

            logit = sess.run(
                scoring_network.outputs['logit'],
                feed_dict={
                    rna_input_ph: rna_input,
                    const_dna_input_ph: const_dna_input,
                    alt_dna_input_ph: alt_dna_input,
                    const_site_pos_ph: const_site_pos_input,
                    alt_site_pos_ph: alt_site_pos_input
                }
            )

            self.assertEqual(logit.shape, (n_outputs, N, k))
