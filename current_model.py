# coding: utf-8
import tensorflow as tf
import numpy as np


class Config(object):
    word2vec_init   = True
    word2vec_size   = 300
    char_embed = 15
    char_hidden = 100
    hidden_size = 300
    batch_size = 30
    learning_rate = 3e-5
    l2_weight = 1e-7
    penal_weight = 1e-4
    dropout = 1.0
    max_grad = 10
    grad_noise = 0.01
    num_classes = 3
    perspectives = 50
    max_sent = 300
    max_word = 20
    filter_sizes = [1,3,5]
    label_list = ['entailment','neutral','contradiction']


# sent_emb:[b,n,d]  sent_mask:[b,n]
def disan(sent_emb, sent_mask, keep_prob):
    fw_encoding = token2token(sent_emb, sent_mask, True, keep_prob)     # [b,n,d]
    bw_encoding = token2token(sent_emb, sent_mask, False, keep_prob)
    sent_encoding = tf.concat([fw_encoding, bw_encoding], 2)
    return source2token(sent_encoding, sent_mask, keep_prob)            # [b,d]

def token2token(sent_emb, sent_mask, is_forward, keep_prob):
    bs, sl = tf.shape(sent_emb)[0], tf.shape(sent_emb)[1]
    col, row = tf.meshgrid(tf.range(sl), tf.range(sl))            # [n,n]
    direction_mask = tf.greater(row, col) if is_forward else tf.greater(col, row)                       # [n,n]
    direction_mask_tile = tf.tile(tf.expand_dims(direction_mask, 0), [bs, 1, 1])     # [b,n,n]
    length_mask_tile = tf.tile(tf.expand_dims(sent_mask, 1), [1, sl, 1])             # [b,1,n] -> [b,n,n]
    attention_mask = tf.logical_and(direction_mask_tile, length_mask_tile)                              # [b,n,n]

    sent_hidden = tf.nn.dropout(tf.layers.dense(sent_emb, Config.hidden_size, tf.nn.elu), keep_prob)   # [b,n,d]
    matching_col = tf.tile(tf.expand_dims(sent_hidden, 1), [1, sl, 1, 1])             # [b,1,n,d] -> [b,n,n,d]
    matching_row = tf.tile(tf.expand_dims(sent_hidden, 2), [1, 1, sl, 1])             # [b,n,1,d] -> [b,n,n,d]
    matching_logit = tf.layers.dense(tf.concat([matching_col, matching_row], 3)/5.0, Config.hidden_size, tf.tanh)     # [b,n,n,d]

    matching_logit += tf.expand_dims(1-tf.cast(attention_mask, tf.float32), -1) * (-1e30)               # [b,n,n,d]
    attention_score = tf.nn.softmax(matching_logit, 2) * tf.expand_dims(tf.cast(attention_mask, tf.float32), -1)    # [b,n,n,d]
    compare_result = tf.reduce_sum(attention_score * matching_col, 2)                                   # [b,n,d]

    fusion_gate = tf.nn.dropout(tf.layers.dense(tf.concat([sent_hidden, compare_result], -1), Config.hidden_size, tf.sigmoid), keep_prob)  # [b,n,d]
    return fusion_gate*sent_hidden + (1-fusion_gate)*compare_result
    # tf.expand_dims(tf.cast(sent_mask, tf.float32) -1)   # [b,n,d]

def source2token(sent_emb, sent_mask, keep_prob):
    map1 = tf.nn.dropout(tf.layers.dense(sent_emb, 2*Config.hidden_size, tf.nn.elu), keep_prob)   # [b,n,d]
    map2 = tf.nn.dropout(tf.layers.dense(map1, 2*Config.hidden_size), keep_prob)   # [b,n,d]
    map2 += tf.expand_dims(1 - tf.cast(sent_mask, tf.float32), -1) * (-1e30)
    attention_score = tf.nn.softmax(map2, 1) * tf.expand_dims(tf.cast(sent_mask, tf.float32), -1)
    return tf.reduce_sum(attention_score * sent_emb, 1)



class Model(object):
    def __init__(self, embedding):
        self.word_embedding = embedding

    def build_model(self):
        with tf.variable_scope("attention_model",initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.p_words = tf.placeholder(tf.int32, [None, None])                    # (b,m)
            self.q_words = tf.placeholder(tf.int32, [None, None])                    # (b,n)

            self.p_length= tf.reduce_sum(tf.sign(self.p_words),1)
            self.q_length= tf.reduce_sum(tf.sign(self.q_words),1)
            self.p_mask = tf.sequence_mask(self.p_length, tf.shape(self.p_words)[1])
            self.q_mask = tf.sequence_mask(self.q_length, tf.shape(self.q_words)[1])

            self.p_chars = tf.placeholder(tf.int32, [None, None, None])            # (b,m,w)
            self.q_chars = tf.placeholder(tf.int32, [None, None, None])            # (b,n,w)
            self.dropout = tf.placeholder(tf.float32)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            with tf.device('/cpu:0'):
                self.embed_matrix = tf.convert_to_tensor(self.word_embedding,dtype=tf.float32)
                self.p_emb = tf.nn.embedding_lookup(self.embed_matrix, self.p_words)      # (b,m,l)
                self.q_emb = tf.nn.embedding_lookup(self.embed_matrix, self.q_words)      # (b,n,l)

            with tf.variable_scope("disan") as scope:
                self.p_disan = disan(self.p_emb, self.p_mask, self.dropout)
            with tf.variable_scope("disanq") as scope:
                self.q_disan = disan(self.q_emb, self.q_mask, self.dropout)
            """
            with tf.variable_scope("encoding_p"):
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.hidden_size),input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.hidden_size),input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                p_outputs, p_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.p_emb, self.p_length, dtype=tf.float32)
                self.p_output = tf.concat(p_outputs, -1)*tf.expand_dims(tf.sequence_mask(self.p_length,tf.shape(self.p_words)[1],tf.float32),-1)   # (b,m,2d)
                self.p_state = tf.concat(p_states, -1)       # (b,2d)
                self.p_max = tf.reduce_max(self.p_output, 1)


            with tf.variable_scope("encoding_q"):
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.hidden_size),input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(Config.hidden_size),input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                q_outputs, q_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.q_emb, self.q_length, dtype=tf.float32)
                self.q_output = tf.concat(q_outputs, -1)*tf.expand_dims(tf.sequence_mask(self.q_length,tf.shape(self.q_words)[1],tf.float32),-1)   # (b,m,2d)
                self.q_state = tf.concat(q_states, -1)       # (b,2d)
                self.q_max = tf.reduce_max(self.q_output, 1)
            """
            with tf.variable_scope("loss"):
                l0 = tf.concat([self.p_disan, self.q_disan], 1)
                l1 = tf.layers.dense(l0, Config.hidden_size, tf.nn.relu)
                l2 = tf.layers.dense(l1, Config.hidden_size, tf.nn.relu)
                self.logits = tf.layers.dense(l2, 3, None)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels,3,dtype=tf.float32), logits=self.logits),-1)
                # self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.labels,3,dtype=tf.float32), self.logits)
                self.train_op = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss)

            # global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(Config.learning_rate, global_step, 3000, 0.8, staircase=True)
            # opt = tf.train.AdamOptimizer(learning_rate)
            # gvs = opt.compute_gradients(self.loss)
            # gvs = [(tf.clip_by_norm(grad,Config.max_grad), val) for grad,val in gvs]
            # gvs = [(tf.add(grad, tf.random_normal(tf.shape(grad),stddev=Config.grad_noise)), val) for grad,val in gvs]
            # self.train_op = opt.apply_gradients(gvs)

    def train_batch(self, sess, batch_data):
        p_words, q_words, p_chars, q_chars, labels = batch_data
        feed = {self.p_words: p_words,
                self.q_words: q_words,
                self.p_chars: p_chars,
                self.q_chars: q_chars,
                self.labels: labels,
                self.dropout:Config.dropout
               }
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed)
        return loss

    def test_batch(self, sess, batch_test):
        p_words, q_words, p_chars, q_chars, labels = batch_test
        feed = {self.p_words: p_words,
                self.q_words: q_words,
                self.p_chars: p_chars,
                self.q_chars: q_chars,
                self.dropout: 1.0
               }
        logits = sess.run(self.logits, feed_dict = feed)
        predict_true = np.sum(np.equal(labels, np.argmax(logits, axis=1)))
        return predict_true
