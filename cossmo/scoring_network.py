import tensorflow as tf
import numpy as np
from .dna_encoder import dna_encoder
from .constants import INTRON_LENGTH_DISTRIBUTION
import tensorflow.contrib.layers as layers

MIN_FLOAT = np.finfo(np.float32).min


class ScoringNetwork(object):
    parameters = {
        'padding': 'SAME',
        'pooling_method': 'max',
        'dropout_keep_prob': 1.,
        'conv_weights_initializer': tf.truncated_normal_initializer(
            stddev=.01),
        'conv_biases_initializer': tf.constant_initializer(1),
        'fc_weights_initializer': tf.truncated_normal_initializer(stddev=.01),
        'lstm_weights_initializer': tf.truncated_normal_initializer(stddev=1),
        'fc_biases_initializer': tf.constant_initializer(.01),
        'batch_normalization': True,
        'lr_decay': False,
        'init_scale_conv': 0.01,
        'init_scale_fc': 0.01,
        'init_scale_LSTM': 1,
        'learning_rate': 0.003
    }

    nodes = {}

    outputs = {}

    def __init__(self, inputs, parameters, features=['intron_length'], trainable=False, prediction=False):

        self.inputs = inputs
        self.trainable = trainable
        self.reuse_vars = not (trainable) and not (prediction)
        self.parameters.update(parameters)
        init_scale_conv = self.parameters.get('init_scale_conv', 0.01)
        init_scale_fc = self.parameters.get('init_scale_fc', 0.01)
        init_scale_lstm = self.parameters.get('init_scale_lstm', 0.1)
        init_scale_fc_final = self.parameters.get('init_scale_fc_final', 0.01)
        self.parameters['conv_weights_initializer'] = tf.truncated_normal_initializer(stddev=init_scale_conv)
        self.parameters['fc_weights_initializer'] = tf.truncated_normal_initializer(stddev=init_scale_fc)
        self.parameters['lstm_weights_initializer'] = tf.truncated_normal_initializer(stddev=init_scale_lstm)
        self.parameters['fc_final_weights_initializer'] = tf.truncated_normal_initializer(stddev=init_scale_fc_final)
        self.dropout_kp_current = tf.get_variable('dropout_current_kp',
                                                  trainable=False,
                                                  initializer=tf.constant(self.parameters['dropout_keep_prob']))
        self.lr = tf.get_variable('learning_rate',
                                  trainable=False,
                                  initializer=tf.constant(self.parameters['learning_rate']))
        self.dropout_kp_new = tf.placeholder(
            tf.float32, shape=[], name="dropout_new_kp")
        self.dropout_update = tf.assign(self.dropout_kp_current, self.dropout_kp_new)
        self.new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")

        self.lr_update = tf.assign(self.lr, self.new_lr)
        self.decay_counter = 0
        self.histograms = []
        self.performance_metrics = {}
        if trainable:
            self.nodes['batch_ss'] = tf.reduce_sum(self.inputs['n_alt_ss'], name='total_ss')
            self.nodes['batch_seq'] = tf.squeeze(tf.slice(
                tf.shape(self.inputs['rna_seq']), [0], [1]), name='total_seq')
            self.performance_metrics['ss_per_sec'] = tf.placeholder(tf.float32, name='ss_per_sec')
            self.performance_metrics['seq_per_sec'] = tf.placeholder(tf.float32, name='seq_per_sec')
            self.performance_metrics['spearman'] = tf.placeholder(tf.float32, name='spearman')
            self.performance_metrics['xy_plot'] = tf.placeholder(tf.float32, name='xy_plot')
            self.performance_metrics['psi_axis_plot'] = tf.placeholder(tf.float32, name='psi_axis_plot')

        assert self.parameters['pooling_method'] in ('avg', 'max')
        if 'intron_length' in features:
            assert 'const_site_position' in inputs
            assert 'alt_ss_position' in inputs

        const_exon_len = self.parameters.get('const_exonic_seq_length',
                                             self.parameters.get('exonic_seq_length', 40))
        const_intron_len = self.parameters.get('const_intronic_seq_length',
                                               self.parameters.get('intronic_seq_length', 40))
        alt_exon_len = self.parameters.get('alt_exonic_seq_length',
                                           self.parameters.get('exonic_seq_length', 40))
        alt_intron_len = self.parameters.get('alt_intronic_seq_length',
                                             self.parameters.get('intronic_seq_length', 40))
        self.parameters['const_dna_seq_length'] = const_exon_len + const_intron_len
        self.parameters['alt_dna_seq_length'] = alt_exon_len + alt_intron_len
        self.parameters['rna_seq_length'] = const_exon_len + alt_exon_len

        # Int32 vector with number of alt SS for each example
        self.nodes['k'] = tf.squeeze(tf.slice(
            tf.shape(self.inputs['rna_seq']), [1], [1]), name='k')

        with tf.name_scope('genomic_features'):
            with tf.name_scope('intron_length'):
                if 'intron_length' in features:
                    # Compute intron length
                    self.nodes['intron_length'] = self._compute_intron_lengths(
                        self.inputs['const_site_position'],
                        self.inputs['alt_ss_position'])

                    # Normalize the intron length by average human intron
                    # length and standard error
                    self.nodes['intron_length_norm'] = (
                                                           tf.to_float(self.nodes['intron_length']) -
                                                           INTRON_LENGTH_DISTRIBUTION['mean']) / \
                                                       INTRON_LENGTH_DISTRIBUTION['se']

        # Do 1-in-4 encoding for all sequences
        with tf.name_scope('scoring_network_inputs'):
            with tf.variable_scope('dna_inputs'):
                self.nodes['rna_encoded'] = \
                    dna_encoder(self.inputs['rna_seq'])
            with tf.variable_scope('dna_inputs', reuse=True):
                self.nodes['constitutive_site_dna_encoded'] = \
                    dna_encoder(self.inputs['const_dna_seq'])
            with tf.variable_scope('dna_inputs', reuse=True):
                self.nodes['alternative_site_dna_encoded'] = \
                    dna_encoder(self.inputs['alt_dna_seq'])

            # Prepare sequence inputs for network input
            rna_seq = tf.reshape(
                self.nodes['rna_encoded'],
                [-1, self.parameters['rna_seq_length'], 4],
                name='rna_input')
            dna_constitutive_seq = tf.reshape(
                self.nodes['constitutive_site_dna_encoded'],
                [-1, self.parameters['const_dna_seq_length'], 4], name='dna_const_input')
            dna_alternative_seq = \
                tf.reshape(
                    self.nodes['alternative_site_dna_encoded'],
                    [-1, self.parameters['alt_dna_seq_length'], 4],
                    name='dna_alt_input')

        if self.parameters.get('residual_net', False):
            convnet_width_alt, alt_out = self._build_res_net(dna_alternative_seq, 'alt_resnet')
            convnet_width_cons, cons_out = self._build_res_net(dna_constitutive_seq, 'cons_resnet')
            convnet_width_rna, rna_out = self._build_res_net(rna_seq, 'rna_resnet')
            cons_out_exp = tf.reshape(tf.tile(tf.expand_dims(cons_out, 1), [1, self.nodes['k'], 1]),
                                      [-1, convnet_width_cons])
            self.conv_output = tf.concat(axis=1, values=[rna_out, alt_out, cons_out_exp],
                                         name='conv_output_activation')
            convnet_width = convnet_width_alt + convnet_width_cons + convnet_width_rna

        else:
            convnet_width = self._build_conv_net(
                dna_constitutive_seq, dna_alternative_seq,
                rna_seq, no_reshape=self.parameters.get('comm_LSTM_alternate', False))
        # Setup alternative Comm LSTM
        if self.parameters.get('comm_LSTM_alternate', False):
            lstm_input = self.conv_output
            seq_len = convnet_width[0]
            seq_dim = convnet_width[1]
            if 'intron_length' in features:
                fc_input = tf.concat(
                    axis=1, values=[
                        tf.tile(tf.expand_dims(
                            tf.reshape(self.nodes['intron_length_norm'], [-1, 1]), -1),
                            [1, 1, seq_dim]), lstm_input]
                )
                seq_len += 1
            lstm_input = fc_input
            lstm_size = self.parameters.get('comm_NN_LSTM_size', 100)
            lstm_forget_bias = self.parameters.get('comm_NN_LSTM_forget_bias', 1.0)
            num_LSTM_layers = self.parameters.get('comm_NN_LSTM_layers', 1)
            single_cell = tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=lstm_forget_bias,
                                                  state_is_tuple=True,
                                                  initializer=self.parameters['lstm_weights_initializer'],
                                                  reuse=self.reuse_vars)

            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.dropout_kp_current)

            cell = tf.contrib.rnn.MultiRNNCell(
                [cell for _ in range(num_LSTM_layers)], state_is_tuple=True)
            outputs, state = tf.nn.dynamic_rnn(cell, lstm_input, scope='Alternative_LSTM', dtype=tf.float32)
            fc_input = tf.reshape(tf.slice(outputs, [0, seq_len - 2, 0], [-1, 1, -1]), [-1, lstm_size])
            input_dim = lstm_size
        else:
            # Setup fully-connected layers
            with tf.variable_scope('scoring_network_fully_connected'):
                # Concatenate the normalized intron length
                fc_input = self.conv_output
                # Width of the concatenated layer
                input_dim = convnet_width

                if 'intron_length' in features:
                    fc_input = tf.concat(
                        axis=1,
                        values=(fc_input,
                                tf.reshape(self.nodes['intron_length_norm'], [-1, 1]))
                    )
                    input_dim += 1

                if self.parameters.get('comm_NN_LSTM', False):
                    lstm_size = self.parameters.get('comm_NN_LSTM_size', 100)
                    lstm_steps = self.parameters.get('comm_NN_LSTM_steps', 3)
                    lstm_forget_bias = self.parameters.get('comm_NN_LSTM_forget_bias', 1.0)
                    fc_input = self.get_comm_NN_module3D(fc_input, lstm_size, self.nodes['k'], self.inputs['n_alt_ss'],
                                                         input_dim,
                                                         self.dropout_kp_current,
                                                         self.parameters['fc_weights_initializer'],
                                                         self.parameters['fc_biases_initializer'],
                                                         self.histograms,
                                                         var_scope='input_comm_module'
                                                         )

                    if self.parameters['batch_normalization']:
                        fc_input = \
                            tf.contrib.layers.batch_norm(fc_input,
                                                         decay=self.parameters.get('batch_norm_decay', 0.9),
                                                         zero_debias_moving_mean=self.parameters.get(
                                                             'batch_norm_zero_debias', False),
                                                         epsilon=1e-5,
                                                         scale=True,
                                                         is_training=self.trainable,
                                                         scope="comm_NN_LSTM"
                                                         )

                    self.histograms.append(tf.summary.histogram("activations/LSTM_input", fc_input))
                    num_LSTM_layers = self.parameters.get('comm_NN_LSTM_layers', 1)
                    fc_input = \
                        self.get_comm_NN_module3D(fc_input, lstm_size, self.nodes['k'], self.inputs['n_alt_ss'],
                                                  lstm_size,
                                                  self.dropout_kp_current,
                                                  self.parameters['fc_weights_initializer'],
                                                  self.parameters['fc_biases_initializer'],
                                                  self.histograms,
                                                  var_scope='Comm_NN_LSTM/loop_function/Comm_NN_LSTM',
                                                  var_reuse=False
                                                  )

                    def loop(prev, i):
                        layer_value = self.get_comm_NN_module3D(prev, lstm_size, self.nodes['k'],
                                                                self.inputs['n_alt_ss'],
                                                                lstm_size,
                                                                self.dropout_kp_current,
                                                                self.parameters['fc_weights_initializer'],
                                                                self.parameters['fc_biases_initializer'],
                                                                self.histograms,
                                                                var_scope='Comm_NN_LSTM',
                                                                var_reuse=True
                                                                )
                        self.histograms.append(tf.summary.histogram('activation/LSTM/layer%d' % i, layer_value))
                        return layer_value

                    single_cell = tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=lstm_forget_bias,
                                                          state_is_tuple=True,
                                                          initializer=self.parameters['lstm_weights_initializer'],
                                                          reuse=self.reuse_vars)
                    cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.dropout_kp_current)
                    cell = tf.contrib.rnn.MultiRNNCell(
                        [cell for _ in range(num_LSTM_layers)], state_is_tuple=True)
                    init_state = cell.zero_state(tf.shape(fc_input)[0], tf.float32)
                    outputs, state = tf.contrib.legacy_seq2seq.rnn_decoder([fc_input for _ in range(lstm_steps)],
                                                                           init_state, cell,
                                                                           loop_function=loop,
                                                                           scope='Comm_NN_LSTM')
                    fc_input = self.get_comm_NN_module3D(outputs[-1], lstm_size, self.nodes['k'],
                                                         self.inputs['n_alt_ss'],
                                                         lstm_size,
                                                         self.dropout_kp_current,
                                                         self.parameters['fc_weights_initializer'],
                                                         self.parameters['fc_biases_initializer'],
                                                         self.histograms,
                                                         var_scope='Comm_NN_LSTM/loop_function/Comm_NN_LSTM',
                                                         var_reuse=True
                                                         )
                    self.histograms.append(tf.summary.histogram("activations/LSTM_output", fc_input))
                    input_dim = lstm_size
                else:
                    for i, hu in enumerate(self.parameters['hidden_units']):
                        if self.parameters.get('comm_NN_fc', False):
                            with tf.variable_scope("Comm_fc_%d" % i):
                                fc_input = self.get_comm_NN_module3D(fc_input, hu, self.nodes['k'],
                                                                     self.inputs['n_alt_ss'],
                                                                     input_dim,
                                                                     self.dropout_kp_current,
                                                                     self.parameters['fc_weights_initializer'],
                                                                     self.parameters['fc_biases_initializer'],
                                                                     self.histograms,
                                                                     var_scope="Comm_fc_%d" % i,
                                                                     )

                                input_dim = hu
                                if self.parameters['batch_normalization']:
                                    fc_input = \
                                        tf.contrib.layers.batch_norm(fc_input,
                                                                     decay=self.parameters.get('batch_norm_decay', 0.9),
                                                                     zero_debias_moving_mean=self.parameters.get(
                                                                         'batch_norm_zero_debias', False),
                                                                     epsilon=1e-5,
                                                                     scale=True,
                                                                     is_training=self.trainable,
                                                                     scope="fc_%d" % i
                                                                     )
                        else:
                            with tf.variable_scope("fc_%d" % i):
                                fc_input = \
                                    self.get_fc_layer(
                                        fc_input, hu, input_dim,
                                        self.dropout_kp_current,
                                        self.parameters['fc_weights_initializer'],
                                        self.parameters['fc_biases_initializer'],
                                        self.histograms)

                                if self.parameters['batch_normalization']:
                                    fc_input = \
                                        tf.contrib.layers.batch_norm(fc_input,
                                                                     decay=self.parameters.get('batch_norm_decay', 0.9),
                                                                     zero_debias_moving_mean=self.parameters.get(
                                                                         'batch_norm_zero_debias', False),
                                                                     epsilon=1e-5,
                                                                     scale=True,
                                                                     is_training=self.trainable,
                                                                     scope="fc_%d" % i
                                                                     )
                                self.histograms.append(tf.summary.histogram("activations", fc_input))
                                input_dim = hu

        # Finally, apply the last fully-connected layer,
        # which outputs a scalar
        if self.parameters.get('output_LSTM', False):
            with tf.variable_scope('output_LSTM'):
                out_cell = tf.contrib.rnn.LSTMCell(self.parameters.get('out_LSTM_size', 10),
                                                   initializer=self.parameters['lstm_weights_initializer'],
                                                   reuse=self.reuse_vars)
                outputs, state = tf.nn.dynamic_rnn(out_cell, tf.reshape(fc_input, [-1, self.nodes['k'], input_dim]),
                                                   dtype=tf.float32,
                                                   sequence_length=self.inputs['n_alt_ss'],
                                                   time_major=False)
                fc_input = tf.reshape(outputs, [-1, self.parameters.get('out_LSTM_size', 10)])
                input_dim = self.parameters.get('out_LSTM_size', 10)
        elif self.parameters.get('output_LSTM_bidirectional', False):
            with tf.variable_scope('output_LSTM'):
                out_cell_fw = tf.contrib.rnn.LSTMCell(self.parameters.get('out_LSTM_size', 10),
                                                      initializer=self.parameters['lstm_weights_initializer'],
                                                      reuse=self.reuse_vars)
                out_cell_bw = tf.contrib.rnn.LSTMCell(self.parameters.get('out_LSTM_size', 10),
                                                      initializer=self.parameters['lstm_weights_initializer'],
                                                      reuse=self.reuse_vars)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(out_cell_fw, out_cell_bw,
                                                                 tf.reshape(fc_input, [-1, self.nodes['k'], input_dim]),
                                                                 dtype=tf.float32,
                                                                 sequence_length=self.inputs['n_alt_ss'],
                                                                 time_major=False)
                outputs = tf.concat(outputs, 2)
                fc_input = tf.reshape(outputs, [-1, 2 * self.parameters.get('out_LSTM_size', 10)])
                input_dim = self.parameters.get('out_LSTM_size', 10) * 2
        self.histograms.append(tf.summary.histogram("fc_final_input", fc_input))
        with tf.variable_scope('fc_final'):
            if self.parameters.get('comm_NN_fc_final', False):
                self.nodes['logit_skinny'] = \
                    self.get_comm_NN_module3D(
                        fc_input,
                        self.parameters['n_outputs'], self.nodes['k'], self.inputs['n_alt_ss'],
                        input_dim,
                        tf.constant(1., tf.float32),
                        self.parameters['fc_final_weights_initializer'],
                        self.parameters['fc_biases_initializer'],
                        self.histograms,
                        activation_function=None
                    )
            else:
                self.nodes['logit_skinny'] = \
                    self.get_fc_layer(
                        fc_input,
                        self.parameters['n_outputs'],
                        input_dim,
                        tf.constant(1., tf.float32),
                        self.parameters['fc_final_weights_initializer'],
                        self.parameters['fc_biases_initializer'],
                        self.histograms,
                        activation_function=None
                    )

            self.histograms.append(tf.summary.histogram("logits", self.nodes['logit_skinny']))
            self.nodes['logit_skinny'] = tf.transpose(
                self.nodes['logit_skinny'],
                name='logit_skinny'
            )

            # Reshape the logits
            self.outputs['logit'] = tf.reshape(
                self.nodes['logit_skinny'],
                tf.stack([self.parameters['n_outputs'], -1,
                          self.nodes['k']]),
                name='logit')
        weight_vars = tf.contrib.framework.get_variables_by_name('weights')
        self.weight_norm = tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in weight_vars]))

    def _build_conv_net(self, dna_constitutive_seq,
                        dna_alternative_seq, rna_seq, no_reshape=False):
        n_input_channels = 4

        pool_fct = tf.nn.avg_pool if \
            self.parameters['pooling_method'] == 'avg' else \
            tf.nn.max_pool

        with tf.name_scope('scoring_network_convnet'):
            for i, cp in enumerate(self.parameters['conv_params']):
                if self.parameters.get('comm_NN_conv', False):
                    with tf.variable_scope("conv_%d_rna" % i):
                        rna_seq = \
                            self.get_conv1d_layer_comm(
                                rna_seq, cp[0], self.nodes['k'], self.inputs['n_alt_ss'],
                                cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_rna_%d' % i
                            )

                        if self.parameters['batch_normalization']:
                            rna_seq = \
                                tf.contrib.layers.batch_norm(rna_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope='conv_rna_%d' % i)
                        rna_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(rna_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_rna_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", rna_seq))
                else:
                    with tf.variable_scope("conv_%d_rna" % i):
                        # Convolve the RNA sequences
                        rna_seq = \
                            self.get_conv1d_layer(
                                rna_seq, cp[0], cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_rna_%d' % i
                            )
                        if self.parameters['batch_normalization']:
                            rna_seq = \
                                tf.contrib.layers.batch_norm(rna_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope='conv_rna_%d' % i)

                        rna_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(rna_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_rna_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", rna_seq))

                if self.parameters.get('comm_NN_conv', False):
                    with tf.variable_scope("conv_%d_const_dna" % i):
                        dna_constitutive_seq = \
                            self.get_conv1d_layer_comm(
                                dna_constitutive_seq, cp[0], 1, self.inputs['n_alt_ss'],
                                cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_dna_const_%d' % i
                            )
                        if self.parameters['batch_normalization']:
                            dna_constitutive_seq = \
                                tf.contrib.layers.batch_norm(dna_constitutive_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope="conv_%d_const_dna" % i
                                                             )
                        dna_constitutive_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(dna_constitutive_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_dna_const_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", dna_constitutive_seq))

                else:
                    with tf.variable_scope("conv_%d_const_dna" % i):
                        # Convolve the constitutive DNA sequence
                        dna_constitutive_seq = \
                            self.get_conv1d_layer(
                                dna_constitutive_seq,
                                cp[0], cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_dna_const_%d' % i)
                        if self.parameters['batch_normalization']:
                            dna_constitutive_seq = \
                                tf.contrib.layers.batch_norm(dna_constitutive_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope="conv_%d_const_dna" % i
                                                             )
                        dna_constitutive_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(dna_constitutive_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_dna_const_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", dna_constitutive_seq))

                if self.parameters.get('comm_NN_conv', False):
                    with tf.variable_scope("conv_%d_alternative_dna" % i):
                        dna_alternative_seq = \
                            self.get_conv1d_layer_comm(
                                dna_alternative_seq, cp[0], self.nodes['k'], self.inputs['n_alt_ss'],
                                cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_dna_alternative_%d' % i
                            )

                        if self.parameters['batch_normalization']:
                            dna_alternative_seq = \
                                tf.contrib.layers.batch_norm(dna_alternative_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope="conv_%d_alternative_dna" % i
                                                             )
                        dna_alternative_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(dna_alternative_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_dna_alternative_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", dna_alternative_seq))


                else:
                    with tf.variable_scope("conv_%d_alternative_dna" % i):
                        # Convolve the alternative DNA sequences
                        dna_alternative_seq = \
                            self.get_conv1d_layer(
                                dna_alternative_seq,
                                cp[0], cp[1],
                                n_input_channels, cp[2],
                                self.parameters['conv_weights_initializer'],
                                self.parameters['conv_biases_initializer'],
                                self.parameters['padding'],
                                'conv_dna_alternative_%d' % i
                            )
                        if self.parameters['batch_normalization']:
                            dna_alternative_seq = \
                                tf.contrib.layers.batch_norm(dna_alternative_seq,
                                                             decay=self.parameters.get('batch_norm_decay', 0.9),
                                                             zero_debias_moving_mean=self.parameters.get(
                                                                 'batch_norm_zero_debias', False),
                                                             epsilon=1e-5,
                                                             scale=True,
                                                             is_training=self.trainable,
                                                             scope="conv_%d_alternative_dna" % i
                                                             )

                        dna_alternative_seq = tf.squeeze(pool_fct(
                            tf.expand_dims(dna_alternative_seq, 1),
                            [1, 1, cp[3], 1],  # ksize
                            [1, 1, cp[4], 1],  # strides
                            self.parameters['padding'],
                            name='pool_dna_alternative_%d' % i
                        ), [1])
                        self.histograms.append(tf.summary.histogram("activations", dna_alternative_seq))

                n_input_channels = cp[1]

            dna_alt_width = dna_alternative_seq._shape_as_list()[1]
            dna_const_width = dna_constitutive_seq._shape_as_list()[1]
            rna_width = rna_seq._shape_as_list()[1]

            if no_reshape:
                self.conv_const_dna_expanded = \
                    tf.reshape(
                        tf.tile(
                            tf.expand_dims(
                                dna_constitutive_seq, 1),
                            tf.stack([1, self.nodes['k'], 1, 1])
                        ),
                        [-1, dna_const_width, n_input_channels],
                        name='tiled_const_dna'
                    )
            else:
                # Repeat the convolved constitutive DNA K times
                self.conv_const_dna_expanded = \
                    tf.reshape(
                        tf.tile(
                            tf.expand_dims(
                                dna_constitutive_seq, 1),
                            tf.stack([1, self.nodes['k'], 1, 1])
                        ),
                        [-1, n_input_channels * dna_const_width],
                        name='tiled_const_dna'
                    )

            if no_reshape:
                self.conv_output = \
                    tf.concat(axis=1,
                              values=[
                                  tf.reshape(
                                      rna_seq,
                                      [-1, rna_width, n_input_channels]
                                  ),
                                  self.conv_const_dna_expanded,
                                  tf.reshape(
                                      dna_alternative_seq,
                                      [-1, dna_alt_width, n_input_channels]
                                  )
                              ],
                              name='conv_output_concat')

                convnet_width = (rna_width + dna_alt_width + dna_const_width, n_input_channels)
            else:
                # Concatenate convnet outputs
                self.conv_output = \
                    tf.concat(axis=1,
                              values=[
                                  tf.reshape(
                                      rna_seq,
                                      [-1, n_input_channels * rna_width]
                                  ),
                                  self.conv_const_dna_expanded,
                                  tf.reshape(
                                      dna_alternative_seq,
                                      [-1, n_input_channels * dna_alt_width]
                                  )
                              ],
                              name='conv_output_concat')
                convnet_width = n_input_channels * (rna_width + dna_const_width + dna_alt_width)
            self.histograms.append(tf.summary.histogram("convnet_output", self.conv_output))
        return convnet_width

    def _build_res_net(self, dna_alternative_seq, scope='scoring_network_resnet'):
        with tf.variable_scope(scope, reuse=self.reuse_vars):
            inp_p = self.parameters['resnet_params'][0]
            resnet_output = layers.convolution2d(dna_alternative_seq, num_outputs=inp_p[1], kernel_size=inp_p[0],
                                                 stride=inp_p[2], scope='resnet_block_input')
            resnet_output = tf.squeeze(tf.nn.max_pool(
                tf.expand_dims(resnet_output, 1),
                [1, 1, inp_p[3], 1],  # ksize
                [1, 1, inp_p[4], 1],  # strides
                self.parameters['padding'],
                name='resnet_pool_1'
            ), [1])
            in_dim = inp_p[1]

            for i, cp in enumerate(self.parameters['resnet_params'][1:]):
                with tf.variable_scope('resnet_block_%d' % i):
                    resnet_output = self.get_bottleneck_block(resnet_output, in_dim=in_dim, out_dim=cp[1], width=cp[0],
                                                              oper_dim=cp[2], stride=cp[3],
                                                              is_training=self.trainable,
                                                              )
                in_dim = cp[1]
            resnet_output = tf.reduce_mean(resnet_output, [1], name='avg_pool')
            return resnet_output._shape_as_list()[1], resnet_output


    def assign_dropout_kp(self, session, kp_value):
        session.run(self.dropout_update, feed_dict={self.dropout_kp_new: kp_value})
        return session.run(self.dropout_kp_current)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})
        return session.run(self.lr)

    @staticmethod
    def get_conv1d_layer(input, filter_width, n_output_channels,
                         n_input_channels, stride,
                         conv_weights_initializer, conv_biases_initializer,
                         padding='SAME', op_name=None, no_non_linearity=False):
        W = tf.get_variable("filters", [filter_width, n_input_channels,
                                        n_output_channels],
                            initializer=conv_weights_initializer)
        b = tf.get_variable("biases", [n_output_channels],
                            initializer=conv_biases_initializer)
        if no_non_linearity:
            output = tf.nn.conv1d(input, W, stride, padding, name=op_name) + b
        else:
            output = tf.nn.relu(
                tf.nn.conv1d(input, W, stride, padding, name=op_name) + b, name='activation')
        return output

    @staticmethod
    def get_fc_layer(input, n_hidden_units, input_dim,
                     dropout_prob, weights_initializer, bias_initializer, histograms,
                     activation_function=tf.nn.relu):
        W = tf.get_variable('weights',
                            [input_dim, n_hidden_units],
                            initializer=weights_initializer)
        b = tf.get_variable('biases',
                            [n_hidden_units],
                            initializer=bias_initializer)

        output = tf.matmul(input, W) + b

        if activation_function:
            output = activation_function(output)
        output = tf.nn.dropout(output, dropout_prob)

        histograms.append(tf.summary.histogram('weights', W))
        return output

    @staticmethod
    def get_comm_NN_module3D(input, n_hidden_units, middle_dim, n_alt_ss,
                             input_dim,
                             dropout_prob, weights_initializer, bias_initializer, histograms,
                             var_scope='Comm_module', var_reuse=False,
                             activation_function=tf.nn.relu):
        with tf.variable_scope(var_scope, reuse=var_reuse):
            W = tf.get_variable('weights',
                                [1, input_dim, n_hidden_units],
                                initializer=weights_initializer)
            W_mean = tf.get_variable('weights_mean',
                                     [1, input_dim, n_hidden_units],
                                     initializer=weights_initializer)
            b = tf.get_variable('biases',
                                [n_hidden_units],
                                initializer=bias_initializer)
            input_3D = tf.reshape(input, [-1, middle_dim, input_dim])
            N = tf.shape(input_3D)[0]
            input_3D_mean = tf.map_fn(
                lambda i: tf.reduce_mean(input_3D[i, :tf.cast(n_alt_ss[i], tf.int32), :], axis=0, keep_dims=True),
                tf.range(N), dtype=tf.float32)
            output = (tf.nn.conv1d(input_3D, W, 1, 'SAME') + b +
                      tf.nn.conv1d(input_3D_mean, W_mean, 1, 'SAME'))

            if activation_function:
                output = activation_function(output)

            histograms.append(tf.summary.histogram("weights", W))
            histograms.append(tf.summary.histogram("weights_mean", W_mean))
            output = tf.reshape(output, [-1, n_hidden_units], name='activation')
            output = tf.nn.dropout(output, dropout_prob, name='post_dropout_activation')
            return output

    @staticmethod
    def get_conv1d_layer_comm(input, filter_width, middle_dim, n_alt_ss,
                              n_output_channels,
                              n_input_channels, stride,
                              conv_weights_initializer, conv_biases_initializer,
                              padding='SAME', op_name=None):
        W = tf.get_variable("filters", [filter_width, n_input_channels,
                                        n_output_channels],
                            initializer=conv_weights_initializer)
        b = tf.get_variable("biases", [n_output_channels],
                            initializer=conv_biases_initializer)
        W_mean = tf.get_variable("filters_mean", [filter_width, n_input_channels,
                                                  n_output_channels],
                                 initializer=conv_weights_initializer)
        input_4D = tf.reshape(input, [-1, middle_dim, input._shape_as_list()[1], n_input_channels])

        N = tf.shape(input_4D)[0]
        input_4D_mean = tf.map_fn(
            lambda i: tf.reduce_mean(input_4D[i, :tf.cast(n_alt_ss[i], tf.int32), :, :], axis=0, keep_dims=False),
            tf.range(N), dtype=tf.float32)
        broadcast_signal = tf.expand_dims(tf.nn.conv1d(input_4D_mean, W_mean, stride, padding), axis=1)
        conv_output = tf.nn.conv1d(input, W, stride, padding) + b
        conv_output = tf.reshape(conv_output, [-1, middle_dim, conv_output._shape_as_list()[1], n_output_channels])
        final_out = tf.nn.relu(conv_output + broadcast_signal)
        final_out = tf.reshape(final_out, [-1, final_out._shape_as_list()[2], n_output_channels], name='activation')
        return final_out

    def _compute_intron_lengths(self, const_site_pos, alt_site_pos):
        """Computes the intron lengths as the distance between
        alternative and constitutive splice sites.
        """

        const_site_pos_tiled = tf.tile(tf.expand_dims(
            const_site_pos, 1),
            tf.stack([1, self.nodes['k']]),
            name='tiled_const_site_pos')

        intron_length = tf.abs(const_site_pos_tiled - alt_site_pos,
                               name='intron_length')
        return intron_length

    @staticmethod
    def get_bottleneck_block(input, in_dim, oper_dim, out_dim, width, stride,
                             is_training,
                             bn_decay=0.9):
        pre_activation = layers.batch_norm(input, activation_fn=tf.nn.relu, scope='preact',
                                           is_training=is_training, decay=bn_decay)

        if out_dim == in_dim:
            assert (stride == 1)
            shortcut = input
        else:
            shortcut = layers.convolution2d(pre_activation, num_outputs=out_dim,
                                            kernel_size=1, stride=stride,
                                            normalizer_fn=None, activation_fn=None,
                                            scope='shortcut'
                                            )
        residual = layers.convolution2d(pre_activation, num_outputs=oper_dim,
                                        kernel_size=1, stride=1, scope='conv1',
                                        normalizer_fn=layers.batch_norm,
                                        normalizer_params={'is_training': is_training, 'decay': bn_decay},

                                        )
        residual = layers.convolution2d(residual, num_outputs=oper_dim,
                                        kernel_size=width, stride=stride, scope='conv2',
                                        normalizer_fn=layers.batch_norm,
                                        normalizer_params={'is_training': is_training, 'decay': bn_decay},

                                        )
        residual = layers.convolution2d(residual, num_outputs=out_dim,
                                        kernel_size=1, stride=1, normalizer_fn=None,
                                        activation_fn=None, scope='conv3',
                                        )
        output = shortcut + residual

        return output


