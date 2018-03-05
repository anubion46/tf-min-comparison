import numpy as np
from random import random, seed
import generator
import tensorflow as tf


def monte_carlo_local(funcs, f, dim, amount):
    seed(12)
    points = [[random() for _ in range(dim)] for _ in range(amount)]
    with tf.Session() as sess:
        old = sess.run(f[0], feed_dict={funcs.x: points[0]})
        index = 0
        for i in range(len(points)):
            new = sess.run(f[0], feed_dict={funcs.x: points[i]})
            if old > new:
                old = new
                index = i
    return old, points[index]


def monte_carlo(functions, dim, amount):
    return [monte_carlo_local(functions, f, dim, amount) for f in functions.generate('c2')]


amount = 15000
dim = 3
functions = generator.FunGen(dim, 1)
print(monte_carlo(functions, dim, amount))
