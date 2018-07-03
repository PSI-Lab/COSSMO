import tensorflow as tf
import numpy as np
import os.path
from cossmo.data_pipeline import convert_to_tf_example, \
    read_single_cossmo_example, dynamic_bucket_data_pipeline
from cossmo.scoring_network import ScoringNetwork
from cossmo.output_networks import RaggedOutputNetwork


SEQ_LEN = 20

def sample_sequence(len):
    return ''.join(['ACGT'[k] for k in np.random.randint(0, 3, len)])


class TestDataPipeline(tf.test.TestCase):

    @staticmethod
    def sample_example(n_alt_ss, event_type=None):
        if event_type is None:
            event_type = ['acceptor', 'donor'][np.random.randint(0, 1)]
        assert event_type in ('acceptor', 'donor')

        const_exonic_seq = sample_sequence(SEQ_LEN)
        const_intronic_seq = sample_sequence(SEQ_LEN)
        alt_exonic_seq = [sample_sequence(SEQ_LEN) for _ in range(n_alt_ss)]
        alt_intronic_seq = [sample_sequence(SEQ_LEN)
                            for _ in range(n_alt_ss)]
        psi_distribution = np.random.rand(n_alt_ss, 1)
        psi_distribution /= psi_distribution.sum(0, keepdims=True)
        psi_std = np.random.rand(n_alt_ss, 1)
        const_site_id = str(np.random.randint(1000))
        const_site_pos = np.random.randint(100000)
        alt_ss_position = np.random.randint(0, 100000, n_alt_ss)
        alt_ss_type = [['annotated', 'gtex', 'maxent', 'hard_negative'][i]
                       for i in np.random.randint(0, 4, n_alt_ss)]

        return alt_exonic_seq, alt_intronic_seq, event_type, \
            const_exonic_seq, const_intronic_seq, n_alt_ss, \
            psi_distribution, psi_std, const_site_id, const_site_pos, \
            alt_ss_position, alt_ss_type

    def write_test_file(self, n_examples, n_alt_ss, event_type=None):
        temp_dir = tf.test.get_temp_dir()
        filename = os.path.join(temp_dir, 'test_records.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        if isinstance(n_alt_ss, int):
            n_alt_ss = n_examples * [n_alt_ss]

        with self.test_session():
            for i, k in zip(range(n_examples), n_alt_ss):
                alt_exonic_seq, alt_intronic_seq, event_type_, \
                const_exonic_seq, const_intronic_seq, n_alt_ss, \
                psi_distribution, psi_std, const_site_id,\
                const_site_pos, alt_ss_position, \
                alt_ss_type = self.sample_example(k, event_type)

                example = convert_to_tf_example(
                    const_exonic_seq, const_intronic_seq,
                    alt_exonic_seq, alt_intronic_seq,
                    psi_distribution, psi_std,
                    alt_ss_position, alt_ss_type,
                    const_site_id, const_site_pos,
                    n_alt_ss, event_type_,
                    None, 'rna1'
                )

                writer.write(example.SerializeToString())
            writer.close()
        return filename

    def test_encoder_decoder(self):
        for strand, coord_sys in [(None, 'rna1'), ('+', 'dna0')]:
            with self.test_session():
                n_alt_ss = np.random.randint(2, 10)
                alt_exonic_seq, alt_intronic_seq, event_type, \
                const_exonic_seq, const_intronic_seq, n_alt_ss, \
                psi_distribution, psi_std, const_site_id, \
                const_site_pos, alt_ss_position, \
                alt_ss_type = self.sample_example(n_alt_ss)

                example = convert_to_tf_example(
                    const_exonic_seq, const_intronic_seq,
                    alt_exonic_seq, alt_intronic_seq,
                    psi_distribution, psi_std,
                    alt_ss_position, alt_ss_type,
                    const_site_id, const_site_pos,
                    n_alt_ss, event_type,
                    strand, coord_sys,
                )

                decoded_example = read_single_cossmo_example(
                    example.SerializeToString(), 1, coord_sys)

                self.assertEqual(decoded_example[0]['n_alt_ss'].eval(), n_alt_ss)
                self.assertEqual(decoded_example[0]['event_type'].eval(),
                                 event_type)
                self.assertEqual(decoded_example[0]['const_seq'][0].eval(),
                                 const_exonic_seq)
                self.assertEqual(decoded_example[0]['const_seq'][1].eval(),
                                 const_intronic_seq)
                self.assertTrue(np.alltrue(
                    [a[0] == b for a, b in zip(
                        decoded_example[1]['alt_seq'].eval(),
                        alt_exonic_seq)]))
                self.assertTrue(np.alltrue(
                    [a[1] == b for a, b in zip(
                        decoded_example[1]['alt_seq'].eval(),
                        alt_intronic_seq)]))
                self.assertTrue(np.allclose(
                    decoded_example[1]['psi'].eval(),
                    psi_distribution))

                if coord_sys == 'dna0':
                    self.assertTrue(decoded_example[0]['strand'].eval(), '+')

    def test_write_to_disk(self):
        n_examples = np.random.randint(100, 200)
        n_alt_ss = np.random.randint(2, 10, n_examples)
        filename = self.write_test_file(n_examples, n_alt_ss)

        with self.test_session() as sess:
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=1)

            reader = tf.TFRecordReader()
            _, serialized_examples = reader.read(filename_queue)

            decoded_features = read_single_cossmo_example(
                serialized_examples, 1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            n_records_read = 0
            try:
                while True:
                    n_alt_ss_val, psi_val = \
                        sess.run([decoded_features[0]['n_alt_ss'],
                                  decoded_features[1]['psi']])
                    self.assertIsInstance(n_alt_ss_val, int)
                    self.assertEqual(n_alt_ss_val, psi_val.shape[0])
                    n_records_read += 1
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()

            coord.join(threads)

            self.assertEqual(n_records_read, n_examples)

    # noinspection PyTypeChecker
    def test_input_data_pipeline(self):
        n_examples = 500
        n_alt_ss = np.random.randint(2, 5, n_examples)

        filename = self.write_test_file(n_examples, n_alt_ss)
        buckets = [0, 2, 3, 4, 5]
        batch_sizes = [10, 7, 5, 4, 2]

        with self.test_session() as sess:
            combined_queue = dynamic_bucket_data_pipeline(
                [filename], 1, buckets, batch_sizes, 'acceptor', 1)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(5):
                val = sess.run(combined_queue)
                k = val['n_alt_ss'][0]
                self.assertTrue(np.all(val['n_alt_ss'] == k))
                self.assertIn(val['n_alt_ss'].shape[0], batch_sizes)
                self.assertEqual(val['alt_seq'].shape[1], k)
                self.assertEqual(val['psi'].shape[2], k)

                self.assertEqual(
                    ''.join(val['alt_dna_seq'][:,:,SEQ_LEN:]
                            .view('c').flatten()),
                    ''.join(val['alt_seq'][:,:,0].flatten())
                )
                self.assertEqual(
                    ''.join(val['alt_dna_seq'][:,:,:SEQ_LEN]
                            .view('c').flatten()),
                    ''.join(val['alt_seq'][:,:,1].flatten())
                )
                self.assertEqual(
                    ''.join(np.squeeze(val['rna_seq'][:,::k,:SEQ_LEN], 1)
                            .view('c').flatten()),
                    ''.join(val['const_seq'][:,0].flatten())
                )
                self.assertEqual(
                    ''.join(val['rna_seq'][:,:,SEQ_LEN:]
                            .view('c').flatten()),
                    ''.join(val['alt_seq'][:,:,0].flatten())
                )
                self.assertEqual(
                    ''.join(val['const_dna_seq'][:,:SEQ_LEN]
                            .view('c').flatten()),
                    ''.join(val['const_seq'][:,0].flatten())
                )
                self.assertEqual(
                    ''.join(val['const_dna_seq'][:,SEQ_LEN:]
                            .view('c').flatten()),
                    ''.join(val['const_seq'][:,1].flatten())
                )

            coord.request_stop()
            coord.join(threads)

    def test_network_input(self):
        n_examples = 500
        n_alt_ss = np.random.randint(2, 5, n_examples)
        batch_size = 10

        filename = self.write_test_file(n_examples, n_alt_ss)
        buckets = [2, 3, 4, 5]

        conv_params = [(3, 2, 1, 3, 1)]
        hidden_units = [10]
        n_outputs = 1

        with self.test_session() as sess:
            mini_batches = dynamic_bucket_data_pipeline(
                [filename], 1, buckets, batch_size, 'acceptor', 1
            )

            network_parameters = {
                'exonic_seq_length': SEQ_LEN,
                'intronic_seq_length': SEQ_LEN,
                'conv_params': conv_params,
                'hidden_units': hidden_units,
                'n_outputs': n_outputs
            }

            scoring_network = ScoringNetwork(
                mini_batches,
                network_parameters,
                features=[]
            )

            cossmo = RaggedOutputNetwork(
                scoring_network.outputs['logit'],
                n_outputs,
                mini_batches['n_alt_ss'],
                1.0,
                {}
            )

            psi_predictions = cossmo.get_psi_predictions()
            cossmo.get_cross_entropy_loss(mini_batches['psi'])
            train_op = cossmo.get_optimizer()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            loss_val = cossmo.loss.eval()

            self.assertIsInstance(loss_val, np.float32)

            coord.request_stop()
            coord.join(threads)
