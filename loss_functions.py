import tensorflow as tf
from random import random, seed


def sum_powers(x, dim):
    seed(10)
    power = 2

    a_random = [random() for _ in range(dim)]
    min_point_random = [random() for _ in range(dim)]
    min_value_random = random()

    a = tf.constant(a_random)
    min_point = tf.constant(min_point_random)
    min_value = tf.constant(min_value_random)

    f = tf.add(min_value, tf.reduce_sum(
        tf.multiply(a, tf.pow(tf.subtract(x, min_point), power))))

    function_look = ''
    for i in range(dim):
        function_look += str(round(a_random[i], 5)) + ' * (x' + str(i + 1) + ' + ' + str(round(min_point_random[i], 5)) + ')^' + str(2) + ' + '
    function_look += str(min_value_random)

    return f, function_look, min_point_random, min_value_random
