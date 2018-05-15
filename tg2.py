import tensorflow as tf
import loss_functions
import matplotlib.pyplot as plt
import os
import numpy as np


def distance(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return np.sqrt(res)/len(x)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dim = 15
epochs = 200
decay_rate = 0.97

x = tf.Variable([1 for _ in range(dim)], name='X', dtype='float32')
global_step = tf.Variable(0, trainable=False)
f = loss_functions.sum_sin(x, dim)


def runSession(starter_learning_rate, step=0):
    if step > 0:
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, step, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0], global_step=global_step)
    else:
        learning_rate = starter_learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0])

    values = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Normalization
        starting_value = sess.run(f[0])
        f_temp = tf.divide(tf.subtract(f[0], tf.constant(f[3])), tf.subtract(tf.constant(starting_value), tf.constant(f[3])))

        for i in range(epochs):
            _, x_new, f_new = sess.run([optimizer, x, f_temp])
            # print(i, 'Difference:\t', round(f_new - f[3], 6))
            values.append(f_new - f[3])

    return values


fig = plt.figure()
plt.plot([i+1 for i in range(epochs)], runSession(0.3, 1), color='red', alpha=0.5)
plt.plot([i+1 for i in range(epochs)], runSession(0.3, 5), color='green', alpha=0.5)
plt.plot([i+1 for i in range(epochs)], runSession(0.3), color='blue', alpha=0.5)

plt.plot([i+1 for i in range(epochs)], [0 for i in range(epochs)], color='black', alpha=0.4)

plt.show()
