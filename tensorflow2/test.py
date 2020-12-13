import numpy as np
import tensorflow as tf

x = np.array(
    [[[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]], [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]])

y = x.mean(axis=(0, 2, 3), keepdims=True)

y_ = x.mean(axis=(0, 2, 3))

print(y)
print(y_)

y__ = x.mean()

print(np.array(5))


# 位置编码
def get_angles(pos, i, d_model):
    angles_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angles_rates


def position_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding


pos_encoding = position_encoding(10, 5)
print(pos_encoding.shape)

print(pos_encoding)

input_word_ids = tf.ragged.constant([[1, 2, 3], [4, 5]], tf.int64)
input_mask = tf.ones_like(input_word_ids).to_tensor()
print(input_mask)
input_mask = tf.zeros_like(input_word_ids).to_tensor()
print(input_mask)

input_mask = tf.ones_like(input_word_ids).to_tensor()
print(input_mask)

print(tf.concat([input_mask, input_word_ids], axis=-1))



