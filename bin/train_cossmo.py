from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime

import matplotlib
import numpy as np
import scipy.stats as stats
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from tensorflow import gfile
open = gfile.Open

##################
# Helper functions
##################


def make_models(train_files, validation_files, test_files,
                configuration, traindir, data_pipeline='dynamic'):
    """Creates two graphs for validation and training, sharing the same variables"""

    import tensorflow as tf
    from cossmo.model import get_model, make_training_model, make_evaluation_model, \
        make_prediction_network

    cossmo_template = tf.make_template('cossmo_model', get_model)

    with tf.name_scope('training'):
        scoring_network_train, output_network_train = make_training_model(
            train_files, configuration, data_pipeline, cossmo_template
        )

    with tf.name_scope('validation'):
        configuration_val = configuration.copy()
        configuration_val['dataset_type'] = 'sequential'
        configuration_val['num_epochs'] = None
        configuration_val['batch_size'] = configuration.get('validation_batch_size', 20)

        scoring_network_val, output_network_val = make_evaluation_model(
            validation_files, configuration_val, 'sequential', cossmo_template
        )

    with tf.name_scope('test'):
        test_configuration = configuration.copy()
        test_configuration['num_epochs'] = 1
        test_configuration['batch_size'] = configuration.get('test_batch_size', 20)

        scoring_network_test, output_network_test = make_evaluation_model(
            test_files, test_configuration, 'sequential', cossmo_template
        )

    # Define the prediction model for saving in a separate graph
    pred_graph = tf.Graph()
    with pred_graph.as_default():

        with tf.variable_scope('cossmo_model'):
            placeholders, scoring_network_pred, output_network_pred = \
                make_prediction_network(configuration)
            psi_predictions = output_network_pred.get_psi_predictions()
            logits = output_network_pred.logits

            for t in placeholders.values():
                tf.add_to_collection('inputs', t)
            tf.add_to_collection('outputs', psi_predictions)
            tf.add_to_collection('outputs', logits)

        tf.train.export_meta_graph(filename=os.path.join(traindir, 'prediction_graph.meta'))

    return scoring_network_train, output_network_train, scoring_network_val, \
           output_network_val, scoring_network_test, output_network_test


def get_conv_filter_visualization(filter_name,
                                  max_images=24,
                                  summary_collections=None):
    """Return a summary op to create the filter vizualization of a conv-filter."""

    import tensorflow as tf

    conv_filters = tf.contrib.framework.get_variables_by_name(
        filter_name)[0]
    summary = tf.summary.image(
        filter_name,
        tf.expand_dims(tf.transpose(conv_filters, [2, 1, 0]), 3),
        max_outputs=max_images,
        collections=summary_collections)
    return summary


########################################
# Main function (runs the training loop)
########################################

def main(configuration=None, continue_from=None,
         intra_op_threads=0, inter_op_threads=0,
         test_only=False):

    import tensorflow as tf
    from cossmo.model import make_summaries
    from cossmo.validation_monitor import ValidationMonitor
    from cossmo.model import make_preset_resnet_architecture

    if continue_from is None:
        assert configuration
    if test_only:
        assert continue_from

    # Use the fold specified in the config file or default to the first fold
    cv_fold = configuration.get('cv_folds', [0])[0]

    if continue_from is None:
        if configuration.get('make_rundir', True):
            traindir = os.path.join(configuration['train_dir'], 'fold{}'.format(cv_fold))
        else:
            traindir = configuration['train_dir']
        if not os.path.exists(traindir):
            os.makedirs(traindir)
        if configuration.get('residual_net_preset', False) and configuration.get('residual_net', False):
            configuration['resnet_params'] = make_preset_resnet_architecture(configuration['residual_net_preset'])

        # Store configuration in traindir
        yaml.dump(
            configuration,
            open(os.path.join(traindir, 'config.yml'), 'w')
        )

    else:
        # Continue training from previous state
        traindir = continue_from
        configuration = yaml.load(open(os.path.join(
            continue_from, 'config.yml'
        )))

    summary_frequency = configuration.get('summary_frequency', 1000)
    # graphing_frequency = configuration.get('graphing_frequency', 10000)

    configuration['n_outputs'] = 1
    datadir = configuration['data_dir']
    dataset_type = configuration['dataset_type']

    assert dataset_type in ('bucketed', 'dynamic', 'sequential')

    cv_files = yaml.load(
        open(os.path.join(datadir, 'cv_splits.yml'))
    )


    ####################################################
    # Get file names for the training and validation set
    ####################################################

    if dataset_type in ('dynamic', 'sequential'):
        train_files = [os.path.join(datadir, f) for f in
                       cv_files[cv_fold]['train']]
        validation_files, train_files = \
            (train_files[:configuration['n_validation_files']],
             train_files[configuration['n_validation_files']:])
    elif dataset_type == 'bucketed':
        train_files = [os.path.join(datadir, f) for f in
                       cv_files[cv_fold]['train']]
        validation_files, train_files = \
            (train_files[:configuration['n_validation_files']],
             train_files[configuration['n_validation_files']:])
    else:
        raise ValueError

    test_files = [os.path.join(datadir, f) for f in cv_files[cv_fold]['test']]

    ###############################################
    # Create the graphs for training and validation
    ###############################################
    main_graph = tf.Graph()
    with main_graph.as_default():
        scoring_network_train, output_network_train, \
        scoring_network_val, output_network_val, \
        scoring_network_test, output_network_test = \
            make_models(train_files, validation_files, test_files,
                        configuration, traindir, dataset_type)

        ################
        # Set up metrics
        ################

        # Training metrics
        model_variables = tf.contrib.framework.get_variables(
            scope='cossmo_model')
        training_variables = tf.contrib.framework.get_variables(
            scope='training')
        trainable_variables = tf.trainable_variables()
        model_variables = list(set(model_variables) | set(trainable_variables))
        for variable in tf.global_variables():
            tf.contrib.framework.add_model_variable(variable)

        saver = tf.train.Saver(model_variables + training_variables,
                               max_to_keep=10,
                               keep_checkpoint_every_n_hours=.5,
                               name='checkpoint_saver')
        best_model_saver = tf.train.Saver(model_variables, name='best_model_saver')

        train_summaries, train_metrics, train_metrics_updates, \
            reset_train_metrics_op = make_summaries(output_network_train, scoring_network_train, 'training')
        current_lr = configuration['learning_rate']

        if configuration.get('conv_params', False):
            train_summaries.extend([
                get_conv_filter_visualization(
                    'conv_0_alternative_dna/filters',
                    configuration.get('max_images', 24),
                    'training_metrics'
                ),
                get_conv_filter_visualization(
                    'conv_0_const_dna/filters',
                    configuration.get('max_images', 24),
                    'training_metrics'
                ),
                get_conv_filter_visualization(
                    'conv_0_rna/filters',
                    configuration.get('max_images', 24),
                    'training_metrics'
                )
            ])

        #########################################
        # Define placeholders for activation maps
        #########################################

        tracking_activations = []
        tracking_activations_tensors = []
        tracking_placeholder_names = []
        graph_summaries = []
        for t in main_graph.get_operations():
            if (t.name.endswith('activation') or t.name.endswith('logit_skinny')) and t.name.startswith('training'):
                tracking_activations.append(t.name)
                tracking_activations_tensors.append(t.name + ':0')
                temp = tf.placeholder(dtype=tf.float32, name=t.name + '_activation_map')
                graph_summaries.append(tf.summary.image(temp.name, temp,
                                                        max_outputs=1,
                                                        collections='activation_maps'))
                tracking_placeholder_names.append(temp.name)

        #########################################
        # Define placeholder for xy scatterplot
        #########################################
        graph_summaries.append(tf.summary.image('PSI pred vs Target',
                                                scoring_network_train.performance_metrics['xy_plot'],
                                                max_outputs=1,
                                                collections='xy_plots'))
        graph_summaries.append(tf.summary.image('PSI pred,target vs axis',
                                                scoring_network_train.performance_metrics['psi_axis_plot'],
                                                max_outputs=1,
                                                collections='xy_plots'))

        # Validation metrics
        val_summaries, val_metrics, val_metrics_updates, \
            reset_val_metrics_op = make_summaries(output_network_val, scoring_network_val, 'validation')

        train_summary_op = tf.summary.merge(train_summaries)
        validation_summary_op = tf.summary.merge(val_summaries)
        graph_summary_op = tf.summary.merge([graph_summaries, train_summary_op])

        # We'll use this tensor to count the number of valuation examples
        validation_step_counter = tf.shape_n([scoring_network_val.inputs['n_alt_ss']])[0]

        # Test metrics
        test_summaries, test_metrics, test_metrics_updates, \
            reset_test_metrics_op = make_summaries(output_network_test, scoring_network_test, 'test')

        test_summary_op = tf.summary.merge(test_summaries)

        #########################################
        # Load file to store performance results
        #########################################

        performance_file = os.path.join(traindir, 'performance.yml')
        if os.path.exists(performance_file):
            performance = yaml.load(open(performance_file))
        else:
            performance = {}

        ###########################
        # Define init_ops and hooks
        ###########################

        init_op = tf.variables_initializer(tf.global_variables())
        local_init_op = tf.variables_initializer(tf.local_variables())

        def init_fn(sess):
            sess.run(reset_train_metrics_op)
            sess.run(reset_val_metrics_op)

        #######################
        # Set up the Supervisor
        #######################

        sv = tf.train.Supervisor(
            logdir=traindir,
            init_op=init_op,
            init_fn=init_fn,
            local_init_op=local_init_op,
            summary_op=0,
            summary_writer=0,
            saver=saver,
            save_model_secs=configuration.get('model_save_interval', 600),
            global_step=output_network_train.global_step
        )
        if not (test_only):
            with sv.managed_session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=intra_op_threads,
                    inter_op_parallelism_threads=inter_op_threads)) as sess:

                train_fetches = {
                    'step': output_network_train.global_step,
                    'train_op': output_network_train.train_op,
                    'misc': train_metrics_updates,
                    'batch_ss': scoring_network_train.nodes['batch_ss'],
                    'batch_seq': scoring_network_train.nodes['batch_seq'],
                    'batch_norm_updates': tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                }
                summary_fetches = {
                    'step': output_network_train.global_step,
                    'psi_prediction': output_network_train.psi_prediction,
                    'psi_targets': output_network_train.psi_targets,
                    'misc': train_metrics_updates,
                }

                total_ss = 0
                total_seq = 0
                start_time = time.time()
                summary_writer = sv.summary_writer
                summary_writer.add_graph(sess.graph)

                # Define the summary function
                def summary_fn(step, train_fetch_vals, start_time):
                    summary_fetch_vals = sess.run(summary_fetches)

                    rho_avg = 0
                    for i, pred in enumerate(summary_fetch_vals['psi_prediction'][0]):
                        rho, p_value = stats.spearmanr(summary_fetch_vals['psi_prediction'][0][i],
                                                       summary_fetch_vals['psi_targets'][0][i])
                        rho_avg += rho
                    rho_avg /= len(summary_fetch_vals['psi_prediction'][0])

                    feeds = {
                        scoring_network_train.performance_metrics['ss_per_sec']: (
                            total_ss / (time.time() - start_time)),
                        scoring_network_train.performance_metrics['seq_per_sec']: (
                            total_seq / (time.time() - start_time)),
                        scoring_network_train.performance_metrics['spearman']:
                            rho_avg}

                    # if not step % graphing_frequency:
                    #     graph_feeds = graphing_fn(step, summary_fetch_vals)
                    #     feeds.update(graph_feeds)

                    #     summary_str = sess.run(fetches=graph_summary_op,
                    #                            feed_dict=feeds)
                    # else:
                    summary_str = sess.run(fetches=train_summary_op,
                                           feed_dict=feeds)

                    print("Step: {}, Loss: {}, Accuracy: {}, Speed: {} SS/s"
                          .format(step, summary_fetch_vals['misc'][0],
                                  summary_fetch_vals['misc'][1],
                                  feeds[scoring_network_train.performance_metrics['ss_per_sec']]))

                    sv.summary_computed(sess, summary_str, step)
                    sess.run(reset_train_metrics_op)

                def graphing_fn(step, summary_fetch_vals):
                    savedir_graphs = os.path.join(traindir, 'figures')
                    if not os.path.exists(savedir_graphs):
                        os.makedirs(savedir_graphs)
                    feed_dict = {}
                    fig = plt.figure(step // summary_frequency)
                    plt.loglog(summary_fetch_vals['psi_prediction'].flatten(),
                               summary_fetch_vals['psi_targets'].flatten(), 'o')
                    plt.title("Predicted vs target values")
                    plt.xlabel('psi_prediction')
                    plt.ylabel('psi_target')
                    plt.grid(True)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.savefig(os.path.join(savedir_graphs, 'xy_plot' + '.png'), dpi=1200, format='png')
                    xy_plot = matplotlib.image.imread(open(os.path.join(savedir_graphs, 'xy_plot' + '.png')))
                    xy_plot = np.expand_dims(xy_plot, 0)
                    feed_dict[scoring_network_train.performance_metrics['xy_plot']] = xy_plot
                    plt.close(fig)

                    fig = plt.figure(step // summary_frequency)
                    plt.semilogy(np.arange(summary_fetch_vals['psi_prediction'].shape[2]),
                                 summary_fetch_vals['psi_prediction'][0][0])
                    plt.semilogy(np.arange(summary_fetch_vals['psi_targets'].shape[2]),
                                 summary_fetch_vals['psi_targets'][0][0])
                    plt.legend(['Prediction', 'Target'])

                    plt.title("PSI vs axis")
                    plt.xlabel('Distance from const_position')
                    plt.ylabel('PSI')
                    plt.grid(True)
                    plt.savefig(os.path.join(savedir_graphs, 'psi_axis_plot' + '.png'), dpi=1200, format='png')
                    xy_plot = matplotlib.image.imread(open(os.path.join(savedir_graphs, 'psi_axis_plot' + '.png')))
                    xy_plot = np.expand_dims(xy_plot, 0)
                    feed_dict[scoring_network_train.performance_metrics['psi_axis_plot']] = xy_plot
                    plt.close(fig)

                    fetches = {}
                    for act in tracking_activations:
                        fetches[act] = act + ':0'
                    activations = sess.run(fetches)

                    for k, v in activations.iteritems():
                        activations[k] = np.reshape(activations[k], (activations[k].shape[0], -1))
                        activations[k] = np.expand_dims(np.transpose(activations[k], [0, 1]), axis=0)
                        activations[k] = np.expand_dims(activations[k], axis=3)
                    max_examples = max([np.shape(act)[1] for _, act in activations.iteritems()])

                    for act_placeholder, act in zip(tracking_placeholder_names, tracking_activations):
                        saving_act = activations[act]
                        if 'const_dna' in act:
                            saving_act = np.tile(saving_act, [1, max_examples // saving_act.shape[1], 1, 1])

                        feed_dict[act_placeholder] = saving_act
                    return feed_dict

                validation_monitor = ValidationMonitor(
                    False, configuration.get('early_stopping_delay', 4),
                    (performance.get('best_validation_loss', np.inf),
                     performance.get('best_validation_step', 0)))

                def validation_fn(step):
                    print("Starting validation...")
                    scoring_network_val.assign_dropout_kp(sess, 1.0)
                    sess.run(reset_val_metrics_op)

                    val_steps = 0

                    while val_steps < configuration['n_validation_steps']:
                        values, batch_size = sess.run((val_metrics_updates, validation_step_counter))
                        val_steps += batch_size[0]
                        loss_val = values[0]
                        accuracy_val = values[1]
                        top_5 = values[2]
                        top_2 = values[3]

                    validation_monitor(loss_val, step)
                    print("Validation accuracy: %.3f, Top-5 accuracy:%.3f, Top-2 accuracy:%.3f, Loss: %.3f" %
                          (accuracy_val, top_5, top_2, loss_val))
                    performance['validation_accuracy'] = float(accuracy_val)
                    performance['validation_loss'] = float(loss_val)
                    performance['validation_step'] = int(step)
                    summary_str = sess.run(validation_summary_op)
                    sv.summary_computed(sess, summary_str, step)

                    if validation_monitor.is_hwm():
                        checkpoint_path = os.path.join(traindir, 'best-model.ckpt')
                        performance['best_validation_accuracy'] = float(accuracy_val)
                        performance['best_validation_loss'] = float(loss_val)
                        performance['best_validation_step'] = int(step)
                        best_model_saver.save(sess, checkpoint_path,
                                              latest_filename='checkpoint-best')
                        print("saving model for best checkpoint")

                    yaml.dump(
                        performance,
                        open(performance_file, 'w'),
                        default_flow_style=False
                    )

                try:

                    ####################
                    # Main training loop
                    ####################

                    while not sv.should_stop() and not validation_monitor.should_stop():
                        scoring_network_train.assign_dropout_kp(sess, configuration.get('dropout_keep_prob', 1.0))
                        current_lr = scoring_network_train.assign_lr(sess, current_lr)
                        train_fetch_vals = sess.run(train_fetches)
                        total_seq += train_fetch_vals['batch_seq']
                        total_ss += train_fetch_vals['batch_ss']
                        step = train_fetch_vals['step']
                        if configuration.get('lr_decay', False):
                            if step in configuration['lr_decay_step']:
                                current_lr = current_lr * configuration.get('lr_decay_rate', 0.5)

                        if not step % summary_frequency:
                            summary_fn(step, train_fetch_vals, start_time)
                            start_time = time.time()
                            total_seq = 0
                            total_ss = 0

                        if not step % configuration['validation_frequency']:
                            validation_fn(step)
                            start_time = time.time()
                        if step > configuration.get('max_sgd_steps', 500000):
                            raise KeyboardInterrupt

                except tf.errors.OutOfRangeError:
                    pass
                except KeyboardInterrupt:
                    print("Training stopped by Ctrl+C.")

                if sv.should_stop():
                    print("Supervisor stopped training.")
                if validation_monitor.should_stop():
                    print("Validation Monitor stopped training.")

                summary_writer.flush()


        with sv.managed_session(config=tf.ConfigProto(
                intra_op_parallelism_threads=intra_op_threads,
                inter_op_parallelism_threads=inter_op_threads)) as sess:
            ################################
            # Evaluate model on test set
            ################################
            try:
                best_model_saver.restore(
                    sess, os.path.join(traindir, 'best-model.ckpt'))
            except tf.errors.NotFoundError:
                print("No best model found")
                return

            test_loss, test_accuracy = (0, 0)
            sess.run(reset_test_metrics_op)
            scoring_network_test.assign_dropout_kp(sess, 1.0)

            with open(
                os.path.join(traindir, 'test_predictions.csv'), 'w'
            ) as test_predictions:
                test_predictions.write('record_key, alt_ss_position, ss_type, '
                                       'logit, psi_prediction, psi_target\n')

                try:
                    while not sv.should_stop():
                        values, record_key, alt_ss_pos, ss_type, \
                        logits, psi_pred, psi_targets = sess.run(
                            [test_metrics_updates,
                             output_network_test.record_key,
                             scoring_network_test.inputs['alt_ss_position'],
                             scoring_network_test.inputs['alt_ss_type'],
                             output_network_test.logits,
                             output_network_test.psi_prediction,
                             output_network_test.psi_targets]
                        )
                        test_loss = values[0]
                        test_accuracy = values[1]

                        for rec, pos, type, logits, psis, targets in \
                            zip(record_key.tolist(),
                                alt_ss_pos.tolist(),
                                ss_type.tolist(),
                                logits[0].tolist(),
                                psi_pred[0].tolist(),
                                psi_targets[0].tolist()):
                            rec = '/'.join(rec.split('/')[-2:])
                            for p, t, logit, psi, target in \
                                zip(pos, type, logits, psis, targets):
                                if p == 0:
                                    break
                                line = "{},{},{},{},{},{}\n".format(
                                    rec, p, t, logit, psi, target)
                                test_predictions.write(line)

                except tf.errors.OutOfRangeError:
                    print("Test set evaluation complete. Loss: {}, Accuracy: {}"
                          .format(test_loss, test_accuracy))
                    performance['test_loss'] = float(test_loss)
                    performance['test_accuracy'] = float(test_accuracy)
                    yaml.dump(
                        performance,
                        open(performance_file, 'w'),
                        default_flow_style=False
                    )
                    summary_str, step = sess.run((test_summary_op,
                                                  output_network_train.global_step))
                    sv.summary_computed(sess, summary_str, step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configuration-file', default=None,
        help="Path to a configuration file, containing all hyperparameters, "
             "model definitions, etc. See the provided examples for details."
    )
    parser.add_argument(
        '--gpu', default=None, type=int,
        help="GPU device ID to use for training. This is equivalent to setting "
             "the CUDA_VISIBLE_DEVICES environment variable. "
             "It is recommend to set this option when you have more than one "
             "GPU device in your system to prevent TensorFlow from claiming "
             "all devices."
    )
    parser.add_argument(
        '--intra-op-threads', default=0, type=int,
        help="See https://github.com/tensorflow/tensorflow/blob/"
             "26b4dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/"
             "protobuf/config.proto#L165 for the meaning of this parameter."
    )
    parser.add_argument(
        '--inter-op-threads', default=0, type=int,
    help="See https://github.com/tensorflow/tensorflow/blob/"
         "26b4dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/"
         "protobuf/config.proto#L165 for the meaning of this parameter."
    )
    parser.add_argument(
        '--test-only', default=False, action='store_true',
        help="Don't train, only evaluate test set."
    )
    parser.add_argument(
        '--fold', default=False, type=int,
        help="Cross-validation fold to train on. When set, this overrides the"
             "`cv_fold` key in the configuration file."
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.configuration_file:
        configuration = yaml.load(open(args.configuration_file))
    else:
        configuration = None

    if args.fold:
        configuration['cv_folds'] = [args.fold]

    main(configuration, None,
         intra_op_threads=args.intra_op_threads,
         inter_op_threads=args.inter_op_threads,
         test_only=args.test_only)
