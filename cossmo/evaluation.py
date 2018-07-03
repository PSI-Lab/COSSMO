import tensorflow as tf
from .data_pipeline import sequential_data_pipeline
from .model import get_ragged_model
from . import get_unitialized_variables
from collections import defaultdict


def evaluate_cossmo_performance(checkpoint, conf, print_progress, sess,
                                test_files):

    batches, cossmo = restore_model(checkpoint, conf, sess, test_files)
    correct = tf.equal(
        tf.argmax(cossmo.psi_prediction, 2),
        tf.argmax(cossmo.psi_targets, 2)
    )
    accuracy_op, accuracy_update = \
        tf.contrib.metrics.streaming_accuracy(
            tf.argmax(cossmo.psi_prediction, 2),
            tf.argmax(cossmo.psi_targets, 2),
            name='test/accuracy'
        )
    loss_op, loss_update = tf.contrib.metrics.streaming_mean(
        cossmo.loss, name='test/loss')
    sess.run(tf.variables_initializer(
        tf.contrib.framework.get_local_variables('test')))
    unitialized_variables = get_unitialized_variables(sess)
    sess.run(tf.variables_initializer(unitialized_variables))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    correct_by_n_alt_ss = defaultdict(list)
    try:
        step = 0
        while not coord.should_stop():
            accuracy, loss, n_alt_ss, correct_val = sess.run(
                [accuracy_update, loss_update,
                 batches['n_alt_ss'], correct])

            for n, c in zip(n_alt_ss, correct_val):
                correct_by_n_alt_ss[n].append(c)

            if print_progress and not step % 1000:
                print step, accuracy, loss

            step += 1

    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
    coord.join(threads)
    return accuracy, correct_by_n_alt_ss, loss


def restore_model(checkpoint, conf, sess, test_files):
    batches = sequential_data_pipeline(
        test_files, 1, 1, 'acceptor', 1)
    cossmo_template = tf.make_template('cossmo_model', get_ragged_model,
                                       create_scope_now_=True)
    scoring_network, cossmo = cossmo_template(
        batches['rna_seq'],
        batches['const_dna_seq'],
        batches['alt_dna_seq'],
        batches['output_mask'],
        conf, trainable=False
    )
    cossmo.get_accuracy(batches['psi'])
    cossmo.get_cross_entropy_loss()
    restore_variables = tf.contrib.framework.get_variables(
        scope='cossmo_model')
    saver = tf.train.Saver(restore_variables)
    saver.restore(sess, checkpoint)
    return batches, cossmo
