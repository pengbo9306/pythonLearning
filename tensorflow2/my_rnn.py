import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time
import math

sys.path.append("..")
print(sys.path)
import d2lzh_tensorflow2 as d2l

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()


# print(tf.one_hot(np.array([0, 1,2]), vocab_size))
def to_onehot(X, size):
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


X = np.arange(10).reshape((2, 5))
input = to_onehot(X, vocab_size)
print((len(input), input[0].shape))

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32))

    # 隐层
    W_xh = _one((num_outputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    # 输出层
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    output = []
    for X in inputs:
        X = tf.reshape(X, (-1, W_xh.shape[0]))
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        output.append(Y)
    return output, (H,)


state = init_rnn_state(X.shape[0], num_hiddens)
# print(state)
input = to_onehot(X, vocab_size)
# print('input: ', input)
params = get_params()
# print('params:', params)
outputs, state_new = rnn(input, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)


# 定义预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = tf.convert_to_tensor(to_onehot(np.array([output[-1]]), vocab_size), dtype=tf.float32)
        X = tf.reshape(X, [1, -1])

        (Y, state) = rnn(X, state, params)

        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y[0], axis=1))))

    return ''.join([idx_to_char[i] for i in output])


print(predict_rnn('分开', 10, rnn, params, init_rnn_state,
                  num_hiddens, vocab_size, idx_to_char, char_to_idx))


# 裁剪梯度
def grad_clipping(grads, theta):
    norm = np.array([0])
    for i in range(len(grads)):
        norm += tf.math.reduce_sum(grads[i] ** 2)
    norm = np.sqrt(norm).item()
    new_gradient = []
    if norm > theta:
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        for grad in grads:
            new_gradient.append(grad)

    return new_gradient


# 定义模型训练函数
def train_and_predict_rnn(
        rnn,
        get_params,
        init_rnn_state,
        num_hiddens,
        vocab_size,
        corpus_indices,
        idx_to_char,
        char_to_idx,
        is_random_iter,
        num_epochs,
        num_steps,
        lr,
        clipping_theta,
        batch_size,
        pred_period,
        pred_len,
        prefixes):
    # 随机采样和连续采样
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    for epoch in range(num_epochs):
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(params)
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size,vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                outputs = tf.concat(outputs, 0)

                y = Y.T.reshape((-1,))
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, outputs))
            grads = tape.gradient(l, params)
            grads = grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, params))

            l_sum += np.array(l).item() * len(y)
            n += len(y)
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f time %.2f sec' %
                  (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(prefix)
                print(' _', predict_rnn(
                    prefix,
                    pred_len,
                    rnn,
                    params,
                    init_rnn_state,
                    num_hiddens,
                    vocab_size,
                    idx_to_char,
                    char_to_idx
                ))


num_epochs, num_steps, batch_size, lr, clipping_theta = 5, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 1, 50, ['分开', '不分开']

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char, char_to_idx, False, num_epochs, num_steps,
                      lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)
