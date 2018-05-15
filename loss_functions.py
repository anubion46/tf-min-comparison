import tensorflow as tf
from random import random
import numpy as np


def sum_powers(x, dim):
    power = 2

    a_random = [random() for _ in range(dim)]
    min_point_random = [random() for _ in range(dim)]
    min_value_random = random()

    a = tf.constant(a_random)
    min_point = tf.constant(min_point_random)
    min_value = tf.constant(min_value_random)

    f = tf.add(min_value, tf.reduce_sum(tf.multiply(a, tf.pow(tf.abs(tf.subtract(x, min_point)), power))))

    function_look = ''
    for i in range(dim):
        function_look += str(round(a_random[i], 5)) + ' * |x' + str(i + 1) + ' - ' + str(round(min_point_random[i], 5)) + '|^' + str(power) + ' + '
    function_look += str(min_value_random)

    return f, function_look, min_point_random, min_value_random


def sum_sin(x, dim):
    a = tf.constant([random() for _ in range(dim)])
    n = tf.constant(float(dim))
    f = tf.divide(tf.abs(tf.reduce_sum(tf.multiply(tf.sin(x), a))), n)
    function_look = 'sum of a_i * sin(x_i) divided by n'
    return f, function_look, [0.0 for _ in range(dim)], 0.0


def csendes(x, dim):
    f = tf.reduce_sum(tf.multiply(tf.pow(x, tf.constant(6.0)), tf.add(tf.constant(2.0), tf.sin(tf.divide(tf.constant(1.0), x)))))
    function_look = 'sum of xi^6 * (2 + sin(1/xi)) from 1 to ' + str(dim)
    return f, function_look, [0.0 for _ in range(dim)], 0.0


def damavandi(x, dim):
    f = tf.multiply(
        tf.subtract(
            tf.constant(1.0), tf.pow(
                tf.abs(tf.divide(tf.multiply(tf.sin(tf.multiply(np.pi, tf.subtract(x[0], tf.constant(2.0)))),
                                             tf.sin(tf.multiply(np.pi, tf.subtract(x[1], tf.constant(2.0))))),
                                 tf.multiply(np.pi**2,
                                             tf.multiply(tf.subtract(x[0], tf.constant(2.0)),
                                                         tf.subtract(x[1], tf.constant(2.0)))))
                       ),
                tf.constant(5.0)
                )
            ),
        tf.add(tf.constant(2.0),
               tf.add(tf.pow(tf.subtract(x[0], tf.constant(7.0)), tf.constant(2.0)),
                      tf.multiply(tf.constant(2.0), tf.pow(tf.subtract(x[1], tf.constant(7.0)), tf.constant(2.0)))))
        )
    function_look = '[1 - |sin[Pi*(x1 - 2)]*sin[Pi*(x2 - 2)]/(Pi^2 * (x1 - 2)*(x2 - 2))|^5]*[2 + (x1 - 7)^2 + 2(x2 - 7)^2]'
    return f, function_look, [2.0 for _ in range(dim)], 0.0


def rastrigin(x, dim):
    f = tf.multiply(
        tf.constant(10.0*dim),
        tf.reduce_sum(
            tf.add(
                tf.subtract(
                    tf.pow(x, 2),
                    tf.multiply(
                        tf.constant(10.0),
                        tf.cos(tf.multiply(tf.constant(2.0*np.pi), x))
                    )
                ),
                tf.constant(10.0)
            )
        )
    )
    function_look = '-'
    return f, function_look, [0.0 for _ in range(dim)], 0.0


def deb01(x, dim):
    f = tf.multiply(
        tf.constant(1.0/dim),
        tf.reduce_sum(
            tf.pow(
                tf.sin(
                    tf.multiply(
                        tf.constant(5.0*np.pi), x)), 6)
        )
    )
    function_look = '-'
    return f, function_look, [0.0 for _ in range(dim)], 0.0


def alpine01(x, dim):
    f = tf.reduce_sum(
        tf.abs(
            tf.add(
                tf.multiply(x, tf.sin(x)),
                tf.multiply(tf.constant(0.1), x))
        )
    )
    function_look = '-'
    return f, function_look, [0.0 for _ in range(dim)], 0.0
