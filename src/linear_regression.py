import tensorflow as tf
import numpy as np


with tf.get_default_graph().as_default():
    x = tf.placeholder(tf.float32, [], name="x")
    y = tf.placeholder(tf.float32, [], name="y")
    k = tf.get_variable("k", shape=[])
    b = tf.get_variable("b", shape=[])
    loss = tf.square(k * x + b - y, name="square_error")

    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_op = optimizer.minimize(loss)


with tf.Session() as sess:
    # Initialize `k` and `b`
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        # y = 3x-1 + noise
        data_x = np.random.random()
        data_y = 3.0 * data_x - 1.0 + 0.0001*np.random.randn()

        k_val, b_val, loss_val, _ = sess.run([k, b, loss, train_op],
                                             feed_dict={x: data_x, y: data_y})
        if i % 20 == 0:
            print("k, b, loss: ", k_val, b_val, loss_val)
