import tensorflow as tf
import generator
from random import random, seed
import numpy as np


seed_counter = 10
radius = 0.7
dec = 0.1
step = 10.0


def training_gradient_descent(losses, m, iterations, learning_rates):
    tf.reset_default_graph()
    gradient_descent_results = {}

    for learning_rate in learning_rates:
        temp = np.empty(0, np.float32)

        for loss in losses.keys():
            for dim in losses[loss][1].keys():
                seed(seed_counter)
                function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                functions = function_generator.generate(losses[loss][0])
                for f in functions:
                    r = radius
                    points = generator.generate_points(dim, m, r, f[2])
                    with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                        global_step = tf.Variable(0.0, name='gradient_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                        if step > 0.0:
                            optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, name='train_gradient')).minimize(f[0], global_step=global_step)
                        else:
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
                            temp = np.append(temp, f_curr)
        gradient_descent_results[learning_rate] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return gradient_descent_results


def training_adam(losses, m, iterations, learning_rates):
    tf.reset_default_graph()
    adam_results = {}
    for learning_rate in learning_rates:
        temp = np.empty(0, np.float32)

        for loss in losses.keys():
            for dim in losses[loss][1].keys():
                seed(seed_counter)
                function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                functions = function_generator.generate(losses[loss][0])
                for f in functions:
                    r = radius
                    points = generator.generate_points(dim, m, r, f[2])

                    with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                        global_step = tf.Variable(0.0, name='adam_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                        if step > 0:
                            optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, name='train_adam')).minimize(f[0], global_step=global_step)
                        else:
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
                            temp = np.append(temp, f_curr)

        adam_results[learning_rate] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return adam_results


def training_momentum(losses, m, iterations, learning_rates):
    tf.reset_default_graph()
    momentum_results = {}
    for learning_rate in learning_rates:
        temp = np.empty(0, np.float32)

        for loss in losses.keys():
            for dim in losses[loss][1].keys():
                seed(seed_counter)

                function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                functions = function_generator.generate(losses[loss][0])
                for f in functions:
                    r = radius
                    points = generator.generate_points(dim, m, r, f[2])

                    with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                        global_step = tf.Variable(0.0, name='momentum_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                        if step > 0:
                            optimizer = tf.train.MomentumOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, name='train_momentum'), momentum=0.999, use_nesterov=True).minimize(f[0], global_step=global_step)
                        else:
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
                            temp = np.append(temp, f_curr)

        momentum_results[learning_rate] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return momentum_results


def training_adagrad(losses, m, iterations, learning_rates):
    tf.reset_default_graph()
    adagrad_results = {}
    for learning_rate in learning_rates:
        temp = np.empty(0, np.float32)

        for loss in losses.keys():
            for dim in losses[loss][1].keys():
                seed(seed_counter)
                function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                functions = function_generator.generate(losses[loss][0])
                for f in functions:
                    r = radius
                    points = generator.generate_points(dim, m, r, f[2])

                    with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                        global_step = tf.Variable(0.0, name='adagrad_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                        if step > 0:
                            optimizer = tf.train.AdagradOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, name='train_adagrad')).minimize(f[0], global_step=global_step)
                        else:
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
                            temp = np.append(temp, f_curr)

        adagrad_results[learning_rate] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return adagrad_results


def training_adadelta(losses, m, iterations, learning_rates, decays):
    tf.reset_default_graph()
    adadelta_results = {}
    for learning_rate in learning_rates:
        adadelta_results[learning_rate] = {}
        for decay in decays:
            temp = np.empty(0, np.float32)

            for loss in losses.keys():
                for dim in losses[loss][1].keys():
                    seed(seed_counter)
                    function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                    functions = function_generator.generate(losses[loss][0])
                    for f in functions:
                        r = radius
                        points = generator.generate_points(dim, m, r, f[2])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='adadelta_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if step > 0:
                                optimizer = tf.train.AdadeltaOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, 'train_adadelta'), rho=decay).minimize(f[0], global_step=global_step)
                            else:
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
                                temp = np.append(temp, f_curr)

            adadelta_results[learning_rate][decay] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return adadelta_results


def training_rms(losses, m, iterations, learning_rates, decays):
    tf.reset_default_graph()
    rms_results = {}
    for learning_rate in learning_rates:
        temp = np.empty(0, np.float32)

        rms_results[learning_rate] = {}
        for decay in decays:
            for loss in losses.keys():
                for dim in losses[loss][1].keys():
                    seed(seed_counter)
                    function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
                    functions = function_generator.generate(losses[loss][0])
                    for f in functions:
                        r = radius
                        points = generator.generate_points(dim, m, r, f[2])

                        with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                            global_step = tf.Variable(0.0, name='rms_train_gs' + str(dim), trainable=False, dtype=tf.float32)
                            if step > 0:
                                optimizer = tf.train.RMSPropOptimizer(learning_rate=tf.train.exponential_decay(learning_rate, global_step, step, dec, name='train_rms'), decay=decay).minimize(f[0], global_step=global_step)
                            else:
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
                                temp = np.append(temp, f_curr)

            rms_results[learning_rate][decay] = {'best': np.nanmin(temp).item(), 'worst': np.nanmax(temp).item(), 'mean': np.nanmean(temp).item(), "median": np.nanmedian(temp).item()}
    return rms_results
