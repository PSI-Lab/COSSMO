import tensorflow as tf
import numpy as np
MIN_FLOAT = np.finfo(np.float32).min


class BalancedOutputNetwork(object):
    def __init__(self,
                 logits,
                 num_outputs,
                 weight_norm,
                 configuration,
                 record_key=None):
        self.logits = logits
        self.num_outputs = num_outputs
        self.weight_norm = weight_norm
        self.configuration = configuration
        self.summaries = []
        if record_key is not None:
            self.record_key = tf.identity(record_key, 'record_key')

    def get_psi_predictions(self):
        self.psi_prediction = tf.map_fn(tf.nn.softmax, self.logits,
                                        name='psi_prediction')

        return self.psi_prediction

    def get_accuracy(self, psi_targets=None):
        if psi_targets is not None:
            self.psi_targets = psi_targets
        self.accuracy = tf.stack(
            [tf.reduce_mean(tf.cast(tf.nn.in_top_k(L, y, 1), tf.float32))
             for L, y in zip(
                tf.unstack(self.psi_prediction +
                           tf.random_uniform(tf.shape(self.psi_targets),
                                             1e-10, 1e-8),
                           self.num_outputs),
                tf.unstack(tf.argmax(self.psi_targets, 2), self.num_outputs)
            )], name='accuracy_matrix'
        )
        self.accuracy = tf.reduce_mean(self.accuracy, name='accuracy')
        return self.accuracy

    def get_top_5(self, psi_targets=None):
        if psi_targets is not None:
            self.psi_targets = psi_targets
        self.top_5_accuracy = tf.stack(
            [tf.reduce_mean(tf.cast(tf.nn.in_top_k(L, y, 5), tf.float32))
             for L, y in zip(
                tf.unstack(self.psi_prediction +
                           tf.random_uniform(tf.shape(self.psi_targets),
                                             0.00000001, .000001), self.num_outputs),
                tf.unstack(tf.argmax(self.psi_targets, 2), self.num_outputs)
            )], name='top_5_matrix'
        )
        self.top_5_accuracy = tf.reduce_mean(self.top_5_accuracy, name='top_5_accuracy')
        return self.top_5_accuracy

    def get_top_2(self, psi_targets=None):
        if psi_targets is not None:
            self.psi_targets = psi_targets
        self.top_2_accuracy = tf.stack(
            [tf.reduce_mean(tf.cast(tf.nn.in_top_k(L, y, 2), tf.float32))
             for L, y in zip(
                tf.unstack(self.psi_prediction +
                           tf.random_uniform(tf.shape(self.psi_targets),
                                             0.00000001, .000001), self.num_outputs),
                tf.unstack(tf.argmax(self.psi_targets, 2), self.num_outputs)
            )], name='top_2_matrix'
        )
        self.top_2_accuracy = tf.reduce_mean(self.top_2_accuracy, name='top_2_accuracy')
        return self.top_2_accuracy

    def get_cross_entropy_loss(self, psi_targets=None):
        if psi_targets is not None:
            self.psi_targets = psi_targets
        self.summaries.append(tf.summary.histogram('psi_targets',self.psi_targets))
        self.summaries.append(tf.summary.histogram('psi_prediction', self.psi_prediction))

        self.softmax_cross_entropy = tf.stack(
            [tf.nn.softmax_cross_entropy_with_logits(logits=L, labels=y)
             for L, y in zip(
                tf.unstack(self.logits, self.num_outputs),
                tf.unstack(self.psi_targets, self.num_outputs)
            )], name='softmax_cross_entropy'
        )
        self.target_entropy = tf.stack(
            [-tf.reduce_sum(L * tf.log(y + 1e-15), reduction_indices=[1])
             for L, y in zip(
                tf.unstack(self.psi_targets, self.num_outputs),
                tf.unstack(self.psi_targets, self.num_outputs)
            )], name='softmax_self_entropy'
        )
        self.prediction_entropy = tf.stack(
            [-tf.reduce_sum(L * tf.log(y + 1e-15), reduction_indices=[1])
             for L, y in zip(
                tf.unstack(self.psi_prediction, self.num_outputs),
                tf.unstack(self.psi_prediction, self.num_outputs)
            )], name='softmax_pred_entropy'
        )
        self.mse = tf.square(self.psi_prediction - self.psi_targets)
        self.prediction_entropy_loss = tf.reduce_mean(self.prediction_entropy, name='pred_ent_loss')
        self.target_entropy_loss = tf.reduce_mean(self.target_entropy, name='target_entropy_loss')
        self.cross_ent_loss = tf.reduce_mean(self.softmax_cross_entropy, name='cross_ent_loss')
        self.mse = tf.reduce_mean(self.mse)
        self.loss = self.cross_ent_loss
        self.loss = tf.check_numerics(self.loss, 'Nan in scalar loss', name='check_numerics')
        self.KL = self.cross_ent_loss - self.target_entropy_loss
        return self.loss

    def get_optimizer(self, *args, **kwargs):
        self.optimizer = tf.train.AdamOptimizer(*args, **kwargs)
        l2_lambda = self.configuration.get('l2_decay', 0)

        max_grad_norm = self.configuration.get('max_grad_norm', np.inf)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss + l2_lambda * self.weight_norm, tvars)
        grads = [tf.clip_by_value(grad, clip_value_min=-100, clip_value_max=100) for grad in grads]
        if 'max_grad_norm' in self.configuration.keys():
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

        self.grad_norm = tf.sqrt(tf.add_n([2 * tf.nn.l2_loss(g) for g in grads]))
        for grad, tvar in zip(grads, tvars):
            if ('filters' in tvar.name) or ('weights' in tvar.name):
                self.summaries.append(tf.summary.histogram(tvar.name + '/gradient', grad))

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self.global_step,
            name='train_op'
        )
        return self.train_op


class RaggedOutputNetwork(BalancedOutputNetwork):
    def __init__(self,
                 logits,
                 num_outputs,
                 n_alt_ss,
                 weight_norm,
                 configuration,
                 record_key=None):
        self.logits = logits
        self.num_outputs = num_outputs
        self.n_alt_ss = n_alt_ss
        self.output_mask = tf.sequence_mask(self.n_alt_ss)
        self.weight_norm = weight_norm
        self.configuration = configuration
        self.summaries = []
        if record_key is not None:
            self.record_key = tf.identity(record_key, 'record_key')

    def get_psi_predictions(self):
        output_mask_tiled = tf.tile(
            tf.expand_dims(self.output_mask, 0),
            [self.num_outputs, 1, 1], name='output_mask_tiled'
        )
        self.masked_logit = tf.where(
            output_mask_tiled, self.logits,
            tf.fill(tf.shape(self.logits), MIN_FLOAT)
        )
        self.psi_prediction = tf.map_fn(tf.nn.softmax, self.masked_logit,
                                        name='psi_prediction')
        return self.psi_prediction
