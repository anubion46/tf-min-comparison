import tensorflow as tf
import generator
from random import random, seed
import numpy as np


seed_counter = 10
radius = 0.7


def training_gradient_descent(losses, m, iterations, learning_rates):
    gradient_descent_results = {}
    for learning_rate in learning_rates:
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

                            for i in range(iterations):
                                _, f_curr = sess.run([optimizer, f_temp])
                            temp.append(f_curr)

        gradient_descent_results[learning_rate] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return gradient_descent_results


def training_adam(losses, m, iterations, learning_rates):
    adam_results = {}
    for learning_rate in learning_rates:
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

                            for i in range(iterations):
                                _, f_curr = sess.run([optimizer, f_temp])
                            temp.append(f_curr)

        adam_results[learning_rate] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return adam_results


def training_momentum(losses, m, iterations, learning_rates):
    momentum_results = {}
    for learning_rate in learning_rates:
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

                            for i in range(iterations):
                                _, f_curr = sess.run([optimizer, f_temp])
                            temp.append(f_curr)

        momentum_results[learning_rate] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return momentum_results


def training_adagrad(losses, m, iterations, learning_rates):
    adagrad_results = {}
    for learning_rate in learning_rates:
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

                            for i in range(iterations):
                                _, f_curr = sess.run([optimizer, f_temp])
                            temp.append(f_curr)

        adagrad_results[learning_rate] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return adagrad_results


def training_adadelta(losses, m, iterations, learning_rates, decays):
    adadelta_results = {}
    for learning_rate in learning_rates:
        adadelta_results[learning_rate] = {}
        for decay in decays:
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

                                for i in range(iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp.append(f_curr)

            adadelta_results[learning_rate][decay] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return adadelta_results


def training_rms(losses, m, iterations, learning_rates, decays):
    rms_results = {}
    for learning_rate in learning_rates:
        temp = []
        rms_results[learning_rate] = {}
        for decay in decays:
            for loss in losses.keys():
                for dim in losses[loss][1].keys():
                    seed(seed_counter)
                    tf.reset_default_graph()
                    function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                    functions = function_generator.generate(losses[loss][0])
                    for f in functions:
                        r = radius
                        points = generator.generate_points(dim, m, r, f[2])

                        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(f[0])

                        for point in points:
                            with tf.Session() as sess:
                                sess.run(tf.global_variables_initializer())
                                sess.run(function_generator.x.assign(point))

                                # Normalization
                                starting_value = sess.run(f[0])
                                f_temp = tf.divide(f[0], tf.constant(starting_value))

                                for i in range(iterations):
                                    _, f_curr = sess.run([optimizer, f_temp])
                                temp.append(f_curr)

            rms_results[learning_rate][decay] = {'best': min(temp).item(), 'worst': max(temp).item(), 'mean': np.asscalar(np.mean(temp)), "median": np.asscalar(np.median(temp))}
    return rms_results
