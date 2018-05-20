import tensorflow as tf
import generator
from random import random, seed
import numpy as np


seed_counter = 10
radius = 0.7
dec = 0.1
step = 10.0


def test_gradient_descent(losses, m, iterations, learning_rate, metric):
    tf.reset_default_graph()
    gradient_descent_results = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='gradient_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0.0:
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_gradient')).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                        gradient_descent_results = np.append(gradient_descent_results, np.array([temp]), axis=0)

    return metric(gradient_descent_results, axis=0)


def test_adam(losses, m, iterations, learning_rate, metric):
    tf.reset_default_graph()
    adam_result = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='adam_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0:
                        optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_adam')).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                        adam_result = np.append(adam_result, np.array([temp]), axis=0)

    return metric(adam_result, axis=0)


def test_momentum(losses, m, iterations, learning_rate, metric):
    tf.reset_default_graph()
    momentum_results = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='momentum_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0:
                        optimizer = tf.train.MomentumOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_momentum'), momentum=0.999, use_nesterov=True).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.999, use_nesterov=True).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                        momentum_results = np.append(momentum_results, np.array([temp]), axis=0)

    return metric(momentum_results, axis=0)


def test_adagrad(losses, m, iterations, learning_rate, metric):
    tf.reset_default_graph()
    adagrad_results = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            tf.reset_default_graph()
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='adagrad_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0:
                        optimizer = tf.train.AdagradOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_adagrad')).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                        adagrad_results = np.append(adagrad_results, np.array([temp]), axis=0)

    return metric(adagrad_results, axis=0)


def test_adadelta(losses, m, iterations, learning_rate, decay, metric):
    tf.reset_default_graph()
    adadelta_results = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = radius
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='adadelta_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0:
                        optimizer = tf.train.AdadeltaOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_adadelta'), rho=decay).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                        adadelta_results = np.append(adadelta_results, np.array([temp]), axis=0)

    return metric(adadelta_results, axis=0)


def test_rms(losses, m, iterations, learning_rate, decay, metric):
    tf.reset_default_graph()
    rms_results = np.empty(shape=(0, iterations), dtype=np.float32)

    for loss in losses.keys():
        for dim in losses[loss][1].keys():
            seed(seed_counter)
            function_generator = generator.FunGen(dim, losses[loss][1][dim], loss)
            functions = function_generator.generate(losses[loss][0])
            for f in functions:
                r = random()
                points = generator.generate_points(dim, m, r, f[2])

                with tf.variable_scope(loss, reuse=tf.AUTO_REUSE):
                    global_step = tf.Variable(0.0, name='rms_test_gs' + str(dim), trainable=False, dtype=tf.float32)
                    if step > 0:
                        optimizer = tf.train.RMSPropOptimizer(learning_rate=tf.train.exponential_decay(learning_rate.astype(np.float32), global_step, step, dec, name='test_rms'), decay=decay).minimize(f[0], global_step=global_step)
                    else:
                        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay).minimize(f[0])

                for point in points:
                    with tf.Session() as sess:
                        sess.run(tf.global_variables_initializer())
                        sess.run(function_generator.x.assign(point))

                        # Normalization
                        starting_value = sess.run(f[0])
                        f_temp = tf.divide(f[0], tf.constant(starting_value))

                        temp = np.empty(iterations)
                        for i in range(iterations):
                            _, f_curr = sess.run([optimizer, f_temp])
                            temp[i] = f_curr

                            rms_results = np.append(rms_results, np.array([temp]), axis=0)

    return metric(rms_results, axis=0)
