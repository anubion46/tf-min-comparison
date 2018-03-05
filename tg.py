import tensorflow as tf
import generator
import numpy as np
import csv
import initial_data as ind


def check_optimizer(optimizer, f, st_p, funcs):
    funcs.x.assign(st_p)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # reset values to wrong
        n = 0
        fs = []
        trend = []
        res_iter = ind.n_iter
        for _ in range(ind.n_iter):
            if len(fs) == ind.window:
                trend.append(np.median(fs))
                del fs[0:ind.window//2]
            _, s = sess.run([optimizer, f[0]])
            fs.append(s)
            n += 1
            if np.abs(fs[-1] - f[2]) <= ind.eps and res_iter > n:
                res_iter = n
    return res_iter, trend


# Identify optimal learning rate of a method for a function
def optimal_learning_rate(method, f, sample_lrs, st_p, funcs, thewriter_t):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        if method == 'momentum':
            min_method = tf.train.MomentumOptimizer(learning_rate=sample, momentum=0.99, use_nesterov=True)
        elif method == 'adam':
            min_method = tf.train.AdamOptimizer(learning_rate=sample)
        elif method == 'adadelta':
            min_method = tf.train.AdadeltaOptimizer(learning_rate=sample)
        elif method == 'adagrad':
            min_method = tf.train.AdagradOptimizer(learning_rate=sample)
        else:
            raise ValueError()

        min_iter = check_optimizer(min_method.minimize(f[0]), f, st_p, funcs)
        with tf.Session() as sess:
            thewriter_t.writerow({'method': method, 'learning_rate': sample, 'value': sess.run(f[0], feed_dict={funcs.x: st_p})})
        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter_t.writerow({'method': method, 'learning_rate': sample, 'value': min_iter[1][i]})

        if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
            res = min_iter
            sample_lr = sample
    return sample_lr, res, st_p


# This is the essentially the most important function. All of the writing to files happens here
def process(function_type, n_funcs, n_points, r, methods):
    # Opening files to write general information for tests and functions
    with open('output/' + function_type + '.csv', 'w', newline='') as output, open('output/functions', 'w', newline='') as output_f:
        # Column names for tests files (i.e. c1, c2)
        fieldnames = ['number', 'start', 'optimal_lr', 'iterations', 'method']
        # Writing columns to file. Look up python's csv module
        thewriter = csv.DictWriter(output, fieldnames=fieldnames)
        thewriter.writeheader()

        # For each dimension in the given dictionary do the following
        for dim in n_funcs.keys():
            # Create a generator object for functions
            funcs = generator.FunGen(dim, n_funcs[dim])

            # Create a required amount of functions of a certain dimension
            for i in range(n_funcs[dim]):
                # Generate a function
                f = funcs.generate(function_type)[i]
                # Generate starting points
                starting_points = generator.PointGen(dim, n_points, r, f[1]).create_points()
                # Counter for points (used to name files)
                l = 0

                # For a file containing info about functions write down the function number
                output_f.write('Function ' + str(i+1) + ' of dimension ' + str(dim) + '\n')
                # Write the generated function to function file
                for h in range(dim):
                    output_f.write(str(round(f[4][h], 5)) + ' * (x' + str(h + 1) + ' + ' + str(round(f[1][h], 5)) + ')^' + str(f[5]) + ' + ')
                output_f.write(str(f[2]) + '\n\n')
                # Write all the generated points to function file
                output_f.write('Starting points:\n')
                for d in range(n_points):
                    output_f.write(str(d + 1) + ': ' + str(starting_points[d]) + '\n')
                output_f.write('\n\n\n')

                # For each starting point do the following
                for st_p in starting_points:
                    # Open file to write down trends
                    with open('output/trend_' + f[3] + '_' + str(l) + '.csv', 'w', newline='') as output_t:
                        # Column names for test files (i.e. trend_*_*)
                        fieldnames_t = ['method', 'learning_rate', 'value']
                        # Writing columns to file. Look up python's csv module
                        thewriter_t = csv.DictWriter(output_t, fieldnames=fieldnames_t)
                        thewriter_t.writeheader()

                        # For each method to be tested do the following
                        for met in methods:
                            # Calculate optimal learning rate
                            res = optimal_learning_rate(met, f, ind.momentum_lr, st_p, funcs, thewriter_t)
                            # Write results of the calculations above to the tests file
                            thewriter.writerow({'number': i, 'start': res[2], 'optimal_lr': res[0], 'iterations': res[1][0], 'method': met})
                    l += 1


def main():
    # Initial approximations
    n_funcs = ind.n_funcs
    n_points = ind.n_points
    r = ind.r
    methods = ind.methods

    # Start the process
    process('c2', n_funcs, n_points, r, methods)


if __name__ == "__main__":
    main()
