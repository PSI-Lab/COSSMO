import tensorflow as tf
from .output_networks import BalancedOutputNetwork, RaggedOutputNetwork
from .scoring_network import ScoringNetwork
from .data_pipeline import make_pipeline, read_from_placeholders


def get_model(
        inputs,
        configuration,
        features=['intron_length'],
        trainable=False,
        output_network='ragged',
        prediction=False):

    assert output_network in ('ragged', 'balanced')

    with tf.name_scope('scoring_network'):
        scoring_network = ScoringNetwork(
            inputs,
            configuration,
            features,
            trainable,
            prediction
        )

    with tf.name_scope('cossmo'):
        if output_network == 'balanced':
            cossmo = BalancedOutputNetwork(
                scoring_network.outputs['logit'],
                configuration['n_outputs'],
                scoring_network.weight_norm,
                configuration,
                record_key=inputs.get('tfrecord_key')
            )
        elif output_network == 'ragged':
            cossmo = RaggedOutputNetwork(
                scoring_network.outputs['logit'],
                configuration['n_outputs'],
                inputs['n_alt_ss'],scoring_network.weight_norm,configuration,
                record_key=inputs.get('tfrecord_key')
            )

        cossmo.get_psi_predictions()

        if trainable:
            cossmo.get_cross_entropy_loss(inputs['psi'])
            cossmo.get_accuracy()
            cossmo.get_top_5()
            cossmo.get_top_2()
            cossmo.get_optimizer(
                learning_rate=scoring_network.lr,
                beta1=configuration.get('beta1', .9),
                beta2=configuration.get('beta2', .999),
                epsilon=configuration.get('epsilon', 1e-8),
                use_locking=configuration.get('use_locking', False)
            )

    return scoring_network, cossmo


def make_training_model(files, configuration, data_pipeline='dynamic', model_fn=get_model):
    assert data_pipeline in ('dynamic', 'bucketed', 'sequential')

    if data_pipeline in ('dynamic', 'sequential'):
        output_network = 'ragged'
    elif data_pipeline == 'bucketed':
        output_network = 'balanced'
    else:
        raise ValueError

    features = configuration.get('features', ['intron_length'])

    with tf.device('/cpu:0'):
        train_batches = make_pipeline(configuration, data_pipeline, files)

    scoring_network, output_network = model_fn(
        train_batches,
        configuration,
        features=features,
        trainable=True,
        output_network=output_network
    )

    return scoring_network, output_network


def make_evaluation_model(files, configuration, data_pipeline='sequential', model_fn=get_model):
    assert data_pipeline in ('dynamic', 'bucketed', 'sequential')

    if data_pipeline in ('dynamic', 'sequential'):
        output_network = 'ragged'
    elif data_pipeline == 'bucketed':
        output_network = 'balanced'
    else:
        raise ValueError

    features = configuration.get('features', ['intron_length'])

    with tf.device('/cpu:0'):
        val_batches = make_pipeline(
            configuration, data_pipeline, files)

    # Sharing weights with training model via template
    scoring_network, output_network = model_fn(
        val_batches,
        configuration,
        features=features,
        trainable=False,
        output_network=output_network
    )
    output_network.get_accuracy(val_batches['psi'])
    output_network.get_cross_entropy_loss()
    output_network.get_top_5()
    output_network.get_top_2()

    return scoring_network, output_network


def make_prediction_network(configuration, output_network='balanced', model_fn=get_model):
    placeholders, model_inputs = read_from_placeholders(
        configuration['event_type'])

    features = configuration.get('features', ['intron_length'])
    scoring_network, output_network = get_model(
        model_inputs, configuration,
        features=features, trainable=False,
        output_network='balanced',
        prediction=True
    )

    return placeholders, scoring_network, output_network


def make_summaries(model, scoring_model, name):
    """Creates a set of summary ops"""

    summaries = []

    # Cross-entropy loss
    loss_avg, loss_update_op = tf.contrib.metrics.streaming_mean(
        model.loss, name='%s/loss_val' % name,
        metrics_collections=['%s_metrics' % name],
        updates_collections=['%s_metrics_updates' % name]
    )
    summaries.append(
        tf.summary.scalar('%s/loss' % name, loss_avg,
                          collections='%s_summaries' % name))

    accuracy_avg, accuracy_update_op = \
        tf.contrib.metrics.streaming_mean(
            model.accuracy,
            name='%s/accuracy_val' % name,
            metrics_collections=['%s_metrics' % name],
            updates_collections=['%s_metrics_updates' % name]
        )
    summaries.append(
        tf.summary.scalar('%s/accuracy' % name, accuracy_avg,
                          collections='%s' % summaries))
    # Top-5 accuracy
    top_5_accuracy_avg, top_5_accuracy_update_op = \
        tf.contrib.metrics.streaming_mean(
            model.top_5_accuracy, name='%s/top_5_accuracy_val' % name,
            metrics_collections=['%s_metrics' % name],
            updates_collections=['%s_metrics_updates' % name]
        )
    summaries.append(
        tf.summary.scalar('%s/top-5-accuracy' % name, top_5_accuracy_avg,
                          collections='%s' % summaries))
    # Top-2 accuracy
    top_2_accuracy_avg, top_2_accuracy_update_op = \
        tf.contrib.metrics.streaming_mean(
            model.top_2_accuracy, name='%s/top_2_accuracy_val' % name,
            metrics_collections=['%s_metrics' % name],
            updates_collections=['%s_metrics_updates' % name]
        )
    summaries.append(
        tf.summary.scalar('%s/top-2-accuracy' % name, top_2_accuracy_avg,
                          collections='%s' % summaries))

    # Pearson correlation
    pearson_avg, pearson_update_op = \
        tf.contrib.metrics.streaming_pearson_correlation(
            model.psi_prediction,model.psi_targets,
            name='%s/pearson_val' % name,
            metrics_collections=['%s_metrics' % name],
            updates_collections=['%s_metrics_updates' % name]
        )
    summaries.append(
        tf.summary.scalar('%s/pearson' % name, pearson_avg,
                          collections='%s' % summaries))
    # Mean-squared loss
    mse_avg, mse_update_op = tf.contrib.metrics.streaming_mean(
        model.mse, name='%s/mse_val' % name,
        metrics_collections=['%s_metrics' % name],
        updates_collections=['%s_metrics_updates' % name]
    )
    summaries.append(
        tf.summary.scalar('%s/mse' % name, mse_avg,
                          collections='%s_summaries' % name))
    # KL
    kl_avg, kl_update_op = tf.contrib.metrics.streaming_mean(
        model.KL, name='%s/KL_val' % name,
        metrics_collections=['%s_metrics' % name],
        updates_collections=['%s_metrics_updates' % name]
    )
    summaries.append(
        tf.summary.scalar('%s/KL' % name, kl_avg,
                          collections='%s_summaries' % name))

    # Cross-entropy loss
    xent_avg, xent_update_op = tf.contrib.metrics.streaming_mean(
        model.cross_ent_loss, name='%s/xent_val' % name,
        metrics_collections=['%s_metrics' % name],
        updates_collections=['%s_metrics_updates' % name]
    )
    summaries.append(
        tf.summary.scalar('%s/xent' % name, xent_avg,
                          collections='%s_summaries' % name))
    # Dropout Prob
    summaries.append(tf.summary.scalar('%s/dropout' % name, scoring_model.dropout_kp_current,
                                       collections='%s_summaries' % name))

    if name == 'training':
        # Learning rate
        summaries.append(tf.summary.scalar('%s/learning_rate' % name, scoring_model.lr,
                                           collections='%s_summaries' % name))
        # Weight norm
        summaries.append(tf.summary.scalar('%s/weight_norm' % name, scoring_model.weight_norm,
                                           collections='%s_summaries' % name))

        # Gradient norm
        summaries.append(tf.summary.scalar('%s/grad_norm' % name, model.grad_norm,
                          collections='%s_summaries' % name))
        # SS per second
        summaries.append(tf.summary.scalar('%s/ss_per_second' % name, scoring_model.performance_metrics['ss_per_sec'],
                                           collections='%s_summaries' % name))
        # Seq per second
        summaries.append(tf.summary.scalar('%s/examples_per_second' % name,
                                           scoring_model.performance_metrics['seq_per_sec'],
                                           collections='%s_summaries' % name))

        # Spearman correlation
        summaries.append(tf.summary.scalar('%s/spearman' % name, scoring_model.performance_metrics['spearman'],
                                           collections='%s_summaries' % name))

        # Histograms
        summaries.append(
            tf.summary.merge(scoring_model.histograms, collections='%s_summaries' % name, name='histograms'))
        # Output model gradients and other summaries
        summaries.append(tf.summary.merge(model.summaries, collections='%s_summaries' % name, name='output_summaries'))

    avg_vars = []
    for n in ['loss', 'accuracy', 'top-5-accuracy', 'top-2-accuracy', 'pearson', 'mse', 'KL', 'xent']:
        avg_vars.extend(
            tf.contrib.framework.get_local_variables('%s/%s' % (name, n)))
    reset_avg_op = tf.variables_initializer(avg_vars)

    metrics_collection = tf.get_collection('%s_metrics' % name)
    update_collection = tf.get_collection('%s_metrics_updates' % name)

    return summaries, metrics_collection, update_collection, reset_avg_op


def make_preset_resnet_architecture(flag='resnet_8'):
    """Returns preset resnet architecture defined as list of bottleneck blocks
        with each block defined as [width,out_dim,oper_dim,stride]
        with oper_dim being the number of filters in the middle of the bottleneck block
        and out_dim being the number of output filters

        The first conv block acts as the input block and is defined as
        [width, filters,stride, pooling_kernel_width, pool_stride]
        """
    conv_1 = [[24, 8, 1, 1, 1]]
    resnet_arch = []
    resnet_arch += conv_1
    if flag == 'resnet_26':
        conv_2 = [[8, 32, 8, 1]] * 1
        conv_3 = [[8, 64, 16, 2]] + [[8, 64, 16, 1]] * 1
        conv_4 = [[8, 128, 32, 2]] + [[8, 128, 32, 1]] * 1
        conv_5 = [[8, 256, 64, 1]] + [[8, 256, 64, 1]] * 1
        resnet_arch += conv_2 + conv_3 + conv_4 + conv_5
    elif flag == 'resnet_14':
        conv_2 = [[8, 32, 8, 2]]
        conv_3 = [[8, 64, 16, 2]]
        conv_4 = [[8, 128, 32, 2]]
        conv_5 = [[8, 256, 64, 1]]
        resnet_arch += conv_2 + conv_3 + conv_4 + conv_5
    elif flag == 'resnet_8':
        conv_2 = [[8, 32, 8, 2]]
        conv_3 = [[8, 64, 16, 2]]
        resnet_arch += conv_2 + conv_3
    elif flag == 'resnet_50':
        conv_2 = [[8, 32, 8, 1]] * 3
        conv_3 = [[8, 64, 16, 2]] + [[8, 64, 16, 1]] * 3
        conv_4 = [[8, 128, 32, 2]] + [[8, 128, 32, 1]] * 5
        conv_5 = [[8, 256, 64, 1]] + [[8, 256, 64, 1]] * 2
        resnet_arch += conv_2 + conv_3 + conv_4 + conv_5
    elif flag == 'resnet_101':
        conv_2 = [[8, 32, 8, 1]] * 3
        conv_3 = [[8, 64, 16, 2]] + [[8, 64, 16, 1]] * 3
        conv_4 = [[8, 128, 32, 2]] + [[8, 128, 32, 1]] * 22
        conv_5 = [[8, 256, 64, 1]] + [[8, 256, 64, 1]] * 2
        resnet_arch += conv_2 + conv_3 + conv_4 + conv_5
    else:
        raise ValueError('Invalid preset for resnet architecture')
    return resnet_arch
