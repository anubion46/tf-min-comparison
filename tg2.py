import tensorflow as tf
import loss_functions
import matplotlib.pyplot as plt
import os
import numpy as np


def distance(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return np.sqrt(res)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dim = 15
epochs = 200
decay_rate = 0.97
starter_learning_rate = 0.3

x = tf.Variable([1 for _ in range(dim)], name='X', dtype='float32')
global_step = tf.Variable(0, trainable=False)
f = loss_functions.csendes(x, dim)


def runSession(step):

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, step, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0], global_step=global_step)

    values = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            _, x_new, f_new, lr_new = sess.run([optimizer, x, f[0], learning_rate])
            print(i, 'Difference:\t', round(f_new - f[3], 6), '\tLR:\t', lr_new)
            values.append(f_new - f[3])
    return values

fig = plt.figure()
plt.plot([i+1 for i in range(epochs)], runSession(1), color='red', alpha=0.5)
plt.plot([i+1 for i in range(epochs)], runSession(5), color='green', alpha=0.5)
plt.plot([i+1 for i in range(epochs)], runSession(epochs), color='blue', alpha=0.5)

plt.plot([i+1 for i in range(epochs)], [0 for i in range(epochs)], color='black', alpha=0.4)

plt.show()
