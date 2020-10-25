import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time
import math

sys.path.append("..")
import d2lzh_tensorflow2 as d2l

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

num_hiddens = 256
cell = keras.layers.SimpleRNNCell(num_hiddens, kernel_initializer='glorot_uniform')
rnn_layer = keras.layers.RNN(cell, time_major=True, return_sequences=True, return_state=True)

batch_size = 2
state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

num_steps = 35
X = tf.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
print(Y.shape)


class RNNModel(keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def get_initial_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)


def predict_rnn_keras(prefix, num_chars, model, vocab_size, idx_to_char, char_to_idx):
    state = model.get_initial_state(batch_size=1, dtype=tf.float32)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        X = np.array(output[-1]).reshape((1, 1))
        Y, state = model(X, state)

        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append((int(np.array(tf.argmax(Y, axis=1)))))

    return ''.join(idx_to_char[i] for i in output)


model = RNNModel(rnn_layer, vocab_size)
print(predict_rnn_keras('分开', 10, model, vocab_size, idx_to_char, char_to_idx))


def train_and_predict_rnn_keras(
        model,
        num_hiddens,
        vocab_size,
        corpus_indices,
        idx_to_char,
        char_to_idx,
        num_epochs,
        num_steps,
        lr,
        clipping_theta,
        batch_size,
        pred_period,
        pred_len,
        prefixes):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    for epoch in range(num_epochs):
        l_sum, n, start = 0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps)
        state = model.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        for X, Y in data_iter:
            with tf.GradientTape(persistent=True) as tape:
                (outputs, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(y, outputs)

            grads = tape.gradient(l, model.variables)
            grads = d2l.grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, model.variables))
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d,perplexity %f,time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start
            ))
            for prefix in prefixes:
                print('_', predict_rnn_keras(
                    prefix,
                    pred_len,
                    model,
                    vocab_size,
                    idx_to_char,
                    char_to_idx))


num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_keras(model, num_hiddens, vocab_size, corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta, batch_size,
                            pred_period, pred_len, prefixes)
