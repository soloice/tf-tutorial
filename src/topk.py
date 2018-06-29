# coding=utf-8

import tensorflow as tf
import numpy as np


BATCH_SIZE, NUM_STEPS, VOCAB_SIZE = 2, 4, 10
HIDDEN_SIZE = 6
TOP_K = 2


a1, a2 = np.random.random([BATCH_SIZE, NUM_STEPS, VOCAB_SIZE]), np.random.random([BATCH_SIZE, NUM_STEPS, HIDDEN_SIZE])
y = np.array([[2, 5, 0, 0], [8, 1, 3, 0]], dtype=np.int32) # [BATCH_SIZE, NUM_STEPS]
print("probabilities: ", a1)
print("cnn_outs:", a2)


probabilities = tf.constant(a1, dtype=tf.float32)
cnn_outs = tf.constant(a2, dtype=tf.float32)
y_inputs = tf.constant(y)

inputs_one_hot = tf.one_hot(indices=y_inputs, depth=VOCAB_SIZE, axis=-1) # [BATCH_SIZE, NUM_STEPS, VOCAB_SIZE]
probs = tf.reduce_max(probabilities * inputs_one_hot, axis=-1, keepdims=False) # [BATCH_SIZE, NUM_STEPS]
zero_mask = tf.cast(tf.equal(y_inputs, tf.zeros_like(y_inputs)), tf.float32)

probs = probs + zero_mask  # 给补零的位置加一（从而结果大于1），防止被取出
# probs = probs * (1 - zero_mask) + tf.ones_like(probs) * zero_mask # 若要严格等价于源代码可以这样写，但是没必要

top_k_values, top_k_indices = tf.nn.top_k(-probs, k=TOP_K) # [BATCH_SIZE, TOP_K], [BATCH_SIZE, TOP_K]

# 形如 [BATCH_SIZE * TOP_K]，第 i * TOP_K 至 (i+1)* TOP_K 之间的元素全是 i。
# 即：[0, 0, ..., 0, 1, 1, ..., 1, ..., BATCH_SIZE-1, BATCH_SIZE-1, ..., BATCH_SIZE-1]
cnn_indices = tf.reshape(tf.tile(tf.reshape(tf.range(BATCH_SIZE), [-1, 1]), multiples=[1, TOP_K]), [-1])
res = tf.transpose(tf.stack([cnn_indices, tf.reshape(top_k_indices, [-1])]), [1, 0])

cnn_outputs = tf.gather_nd(params=cnn_outs, indices=res) # [BATCH_SIZE * TOP_K, HIDDEN_SIZE]
cnn_outputs = tf.reshape(cnn_outputs, [BATCH_SIZE, TOP_K, HIDDEN_SIZE])


with tf.Session() as sess:
    print(probs.eval())
    print(cnn_indices.eval())
    print(top_k_indices.eval())
    print(res.eval())
    print(cnn_outputs.eval())
