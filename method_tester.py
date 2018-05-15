import tensorflow as tf
import generator
from random import random, seed
import numpy as np


seed_counter = 10
radius = 0.7


def test_gradient_descent(losses, m, iterations, learning_rate, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)

    gradient_descent_results = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        gradient_descent_results.append(metric(temp3).item())

    return gradient_descent_results


def test_adam(losses, m, iterations, learning_rate, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)

    adam_result = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        adam_result.append(metric(temp3).item())

    return adam_result


def test_momentum(losses, m, iterations, learning_rate, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.999, use_nesterov=True).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)
    momentum_results = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        momentum_results.append(metric(temp3).item())

    return momentum_results


def test_adagrad(losses, m, iterations, learning_rate, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)
    adagrad_results = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        adagrad_results.append(metric(temp3).item())

    return adagrad_results


def test_adadelta(losses, m, iterations, learning_rate, decay, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)
    adadelta_results = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        adadelta_results.append(metric(temp3).item())

    return adadelta_results


def test_rms(losses, m, iterations, learning_rate, decay, metric):
    temp = []
    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = random()
                points = generator.generate_points(dim, m, r, f[2])

                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp2 = []
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp2.append(f_curr)
                        temp.append(temp2)
    rms_results = []
    for i in range(iterations):
        temp3 = []
        for j in range(len(temp)):
            temp3.append(temp[j][i])
        rms_results.append(metric(temp3).item())

    return rms_results
