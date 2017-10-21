# coding: utf-8
# v1.0 first usable
# v1.1 better reusing
# v1.2 solve full gpu memory
# v1.3 using paper setting, add saver
# v1.4 actually reduce gpu memory use, looking to original implementation
import tensorflow as tf
import numpy as np


class Config(object):
    word2vec_init   = True
    word2vec_size   = 300
    hidden_size = 300
    batch_size = 30
    learning_rate = 0.5
    l2_weight = 5e-5
    dropout = 0.75
    max_sent = 100
    max_word = 20
    label_list = ['entailment','neutral','contradiction']
    restore = False

    char_embed = 15
    char_hidden = 100
    max_grad = 10
    grad_noise = 0.01

# sent_hidden:[b,n,d]  sent_mask:[b,n,1]
def disan(sent_hidden, sent_mask, keep_prob):
    fw_encoding = token2token(sent_hidden * sent_mask, sent_mask, True, keep_prob)     # [b,n,d]
    bw_encoding = token2token(sent_hidden * sent_mask, sent_mask, False, keep_prob)
    sent_encoding = tf.concat([fw_encoding, bw_encoding], 2)
    return source2token(sent_encoding, sent_mask, keep_prob)            # [b,d]

def token2token(sent_hidden, sent_mask, is_forward, keep_prob):
    with tf.variable_scope('forward' if is_forward else 'backward'):
        bs, sl = tf.shape(sent_hidden)[0], tf.shape(sent_hidden)[1]
        col, row = tf.meshgrid(tf.range(sl), tf.range(sl))            # [n,n]
        direction_mask = tf.greater(row, col) if is_forward else tf.greater(col, row)                       # [n,n]
        direction_mask_tile = tf.tile(tf.expand_dims(direction_mask, 0), [bs, 1, 1])     # [b,n,n]
        length_mask_tile = tf.tile(tf.expand_dims(tf.squeeze(tf.cast(sent_mask,tf.bool),-1), 1), [1, sl, 1])             # [b,1,n] -> [b,n,n]
        attention_mask = tf.cast(tf.logical_and(direction_mask_tile, length_mask_tile), tf.float32)         # [b,n,n]

        head = tf.expand_dims(dense(sent_hidden, Config.hidden_size, tf.identity, 1.0, 'head'),1) # [b,1,n,d]
        tail = tf.expand_dims(dense(sent_hidden, Config.hidden_size, tf.identity, 1.0, 'tail'),2) # [b,n,1,d]
        matching_logit = 5.0*tf.tanh((head+tail)/5.0) + tf.expand_dims(1-attention_mask, -1) * (-1e30)     # [b,n,n,d]

        attention_score = tf.nn.softmax(matching_logit, 2) * tf.expand_dims(attention_mask, -1)     # [b,n,n,d]
        compare_result = tf.reduce_sum(attention_score * tf.expand_dims(sent_hidden ,1), 2)         # [b,n,d]

        fusion_gate = dense(tf.concat([sent_hidden, compare_result], 2), Config.hidden_size, tf.sigmoid, keep_prob, 'fusion_gate')  # [b,n,d]
        return (fusion_gate*sent_hidden + (1-fusion_gate)*compare_result) * sent_mask   # [b,n,d]

def source2token(sent_encoding, sent_mask, keep_prob):
    with tf.variable_scope('source2token'):
        map1 = dense(sent_encoding, 2*Config.hidden_size, tf.nn.elu, keep_prob, 'map1')   # [b,n,d]
        map2 = dense(map1, 2*Config.hidden_size, tf.identity, keep_prob, 'map2')   # [b,n,d]
        map2 += -1e30 * (1 - sent_mask)
        attention_score = tf.nn.softmax(map2, 1) * sent_mask
        return tf.reduce_sum(attention_score * sent_encoding, 1)

def dense(input, out_size, activation, keep_prob, scope):
    with tf.variable_scope(scope):
        W = tf.get_variable('W', [input.get_shape()[-1], out_size], dtype=tf.float32)
        b = tf.get_variable('b', [out_size], dtype=tf.float32)
        flatten = tf.matmul(tf.reshape(input, [-1, tf.shape(input)[-1]]), W) + b
        out_shape = [tf.shape(input)[i] for i in range(len(input.get_shape())-1)] + [out_size]
        return tf.nn.dropout(activation(tf.reshape(flatten, out_shape)), keep_prob)

class Model(object):
    def __init__(self, embedding):
        self.word_embedding = embedding

    def build_model(self):
        with tf.variable_scope("attention_model",initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.p_words = tf.placeholder(tf.int32, [None, None])                    # (b,m)
            self.q_words = tf.placeholder(tf.int32, [None, None])                    # (b,n)

            self.p_length= tf.reduce_sum(tf.sign(self.p_words),1)
            self.q_length= tf.reduce_sum(tf.sign(self.q_words),1)
            self.p_mask = tf.expand_dims(tf.sequence_mask(self.p_length, tf.shape(self.p_words)[1], tf.float32), -1)
            self.q_mask = tf.expand_dims(tf.sequence_mask(self.q_length, tf.shape(self.q_words)[1], tf.float32), -1)

            self.p_chars = tf.placeholder(tf.int32, [None, None, None])            # (b,m,w)
            self.q_chars = tf.placeholder(tf.int32, [None, None, None])            # (b,n,w)
            self.dropout = tf.placeholder(tf.float32)
            self.labels = tf.placeholder(tf.int32, [None])                          # (b,)

            with tf.device('/cpu:0'):
                self.embed_matrix = tf.convert_to_tensor(self.word_embedding,dtype=tf.float32)
                self.p_emb = tf.nn.embedding_lookup(self.embed_matrix, self.p_words)      # (b,m,l)
                self.q_emb = tf.nn.embedding_lookup(self.embed_matrix, self.q_words)      # (b,n,l)

            with tf.variable_scope("embedding") as scope:
                self.p_inp = dense(self.p_emb, Config.hidden_size, tf.nn.elu, self.dropout, 'hidden')   # [b,n,d]
                scope.reuse_variables()
                self.q_inp = dense(self.q_emb, Config.hidden_size, tf.nn.elu, self.dropout, 'hidden')   # [b,n,d]


            with tf.variable_scope("disan") as scope:
                self.p_disan = disan(self.p_inp, self.p_mask, self.dropout)
                scope.reuse_variables()
                self.q_disan = disan(self.q_inp, self.q_mask, self.dropout)

            with tf.variable_scope("loss"):
                l0 = tf.concat([self.p_disan, self.q_disan, self.p_disan-self.q_disan, self.p_disan*self.q_disan], 1)
                l1 = dense(l0, Config.hidden_size, tf.nn.elu, self.dropout, 'l1')
                l2 = dense(l1, Config.hidden_size, tf.nn.elu, self.dropout, 'l2')
                self.logits = dense(l2, 3, tf.identity, 1.0, 'logits')
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels,3,dtype=tf.float32), logits=self.logits),-1)
                for v in tf.trainable_variables():
                    self.loss += Config.l2_weight * tf.nn.l2_loss(v)
                self.train_op = tf.train.AdadeltaOptimizer(Config.learning_rate).minimize(self.loss)

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
