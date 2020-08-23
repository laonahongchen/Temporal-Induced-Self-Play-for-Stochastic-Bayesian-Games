import tensorflow as tf


class Predictor(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, pr_space, num_steps, rnn_cell="gru", num_units=32):
        raise NotImplementedError
        # if type(pr_space) != list
        #     pr_space = [pr_space]
        #
        # ob = tf.placeholder(name="ob", dtype=tf.float32, shape=[None, num_steps] + list(ob_space.shape))
        #
        # with tf.variable_scope("rnn"):
        #     if rnn_cell == "gru":
        #         cell = tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=U.normc_initializer(1.0))
        #     elif rnn_cell == "lstm":
        #         cell = tf.nn.rnn_cell.LSTMCell(num_units, kernel_initializer=U.normc_initializer(1.0))
        #     else:
        #         raise NotImplementedError
        #
        #     rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True, unroll=True, input_shape=ob_space.shape,
        #                                     input_length=num_steps)
        #     pr_params = rnn_layer(ob)
        #
        # pr_logits = []
        # self.pr_true = pr_true = []
        # losses = []
        # for i, pr_len in enumerate(pr_space):
        #     pr_logit = tf.layers.dense(pr_params, pr_len, name='pr%d' % i, kernel_initializer=U.normc_initializer(1.0))
        #     pr_true = tf.placeholder(dtype=tf.int32, shape=[None], name='pr_true%d' % i)
        #     pr_true_onehot = tf.one_hot(pr_true, depth=pr_len, axis=-1)
        #     loss = tf.losses.softmax_cross_entropy(onehot_labels=pr_true_onehot, logits=pr_logits,
        #                                            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        #     pr_logits.append(pr_logit)
        #     pr_true.append(pr_true)
        #     losses.append(loss)
        #
        # loss = tf.reduce_sum(losses)