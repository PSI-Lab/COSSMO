"""Builds a test model that always outputs logits of 1 and
uniform PSI distribution."""

import tensorflow as tf
from cossmo.data_pipeline import read_from_placeholders
import os
import yaml

_file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(_file_dir, '../test_data/test_model')


def main(alt_ss_type):
    config = {
        'const_intronic_seq_length': 40,
        'const_exonic_seq_length': 40,
        'alt_intronic_seq_length': 40,
        'alt_exonic_seq_length': 40
    }

    config_path = os.path.join(model_dir, alt_ss_type, 'config.yml')
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    yaml.dump(
        config, open(config_path, 'w'))

    prediction_graph = tf.Graph()
    with prediction_graph.as_default():
        with tf.variable_scope('cossmo_model'):
            placeholders, model_inputs = read_from_placeholders(alt_ss_type)

            dummy_var = tf.Variable(0, dtype=tf.float32)

            logits = tf.expand_dims(
                tf.ones_like(model_inputs['alt_ss_position'], tf.float32), 0) + \
                dummy_var

            psi = tf.nn.softmax(logits)

            for p in placeholders.itervalues():
                tf.add_to_collection('inputs', p)

            tf.add_to_collection('outputs', psi)
            tf.add_to_collection('outputs', logits)

            tf.train.export_meta_graph(
                os.path.join(model_dir, alt_ss_type, 'prediction_graph.meta')
            )

            saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.save(
                session,
                os.path.join(model_dir, alt_ss_type, 'best-model.ckpt')
            )

if __name__ == '__main__':
    for alt_ss_type in ('acceptor', 'donor'):
        main(alt_ss_type)
