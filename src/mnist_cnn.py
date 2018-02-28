import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(x, out_dim=10, scope="fc"):
    with tf.variable_scope(scope):
        in_dim = int(x.get_shape()[-1])
        W_fc = weight_variable([in_dim, out_dim])
        b_fc = bias_variable([out_dim])
        fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)
        return fc


def conv_and_pool(x, kernel_size=5, out_channel=32, scope="conv_layer"):
    with tf.variable_scope(scope):
        in_channel = int(x.get_shape()[-1])
        # print(type(in_channel), in_channel)
        # print(x.get_shape())
        W_conv = weight_variable([kernel_size, kernel_size, in_channel, out_channel])
        b_conv = bias_variable([out_channel])

        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x2(h_conv)
        return h_pool


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)
print(mnist.test.images.shape)

with tf.variable_scope("CNN"):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1_out = conv_and_pool(x_image, kernel_size=5, out_channel=32, scope="conv1")
    conv2_out = conv_and_pool(conv1_out, kernel_size=5, out_channel=64, scope="conv2")

    conv2_out_flat = tf.reshape(conv2_out, [-1, 7 * 7 * 64])
    keep_prob = tf.placeholder(tf.float32)

    fc1 = fc_layer(conv2_out_flat, out_dim=1024, scope="fc1")
    fc1_drop = tf.nn.dropout(fc1, keep_prob=keep_prob)
    fc2 = fc_layer(fc1_drop, out_dim=10, scope="fc2")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y_, 1))
n_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

t1 = time.time()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        t2 = time.time()
        batch = mnist.validation.next_batch(50)
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g, time eclipsed %g" % (i, train_accuracy, t2 - t1))
        t1 = time.time()

total_correct = 0.0
for i in range(200):
    batch = mnist.test.next_batch(50)
    total_correct += n_correct.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy %g" % (total_correct/10000))
