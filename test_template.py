import tensorflow as tf
import generator
import numpy as np
import csv
import os
import itertools


def distance(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return np.sqrt(res)


class Test:
    def __init__(self, path, proto_function, function_amount, m, r, methods, iter_threshold, decay, eps=1e-5):
        self.path = path
        self.proto_function = proto_function
        self.function_amount = function_amount
        self.m = m
        self.r = r
        self.methods = methods
        self.iter_threshold = iter_threshold
        self.eps = eps
        self.decay = decay

    def __check_optimizer(self, optimizer, f, st_p, pre_generated_functions):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # reset values to wrong
            sess.run(pre_generated_functions.x.assign(st_p))

            in_between_points = []
            len_h = []
            res_iter = self.iter_threshold
            for n in range(self.iter_threshold):
                _, s, x = sess.run([optimizer, f[0], pre_generated_functions.x])
                in_between_points.append(s)
                if n > 0:
                    len_h.append(distance(f[2], x))
                else:
                    len_h.append(distance([0 for _ in range(len(x))], x))
                if np.abs(in_between_points[-1] - f[3]) <= self.eps and res_iter > n:
                    res_iter = n

            res_point, res_value = sess.run([pre_generated_functions.x, f[0]])
        return res_iter, in_between_points, res_point, res_value, len_h

    # Identify optimal learning rate of a method for a function
    def __optimal_learning_rate(self, method, f, sample_lrs, st_p, pre_generated_functions, thewriter_t):
        res, sample_lr = [-1, -1], -1
        for sample, decay in itertools.product(sample_lrs, self.decay):
            if method == 'momentum':
                decay_val = '-'
                min_method = tf.train.MomentumOptimizer(learning_rate=sample, momentum=0.99999999, use_nesterov=True)
            elif method == 'adam':
                decay_val = '-'
                min_method = tf.train.AdamOptimizer(learning_rate=sample)
            elif method == 'adadelta':
                decay_val = decay
                min_method = tf.train.AdadeltaOptimizer(learning_rate=sample, rho=decay)
            elif method == 'adagrad':
                decay_val = '-'
                min_method = tf.train.AdagradOptimizer(learning_rate=sample)
            elif method == 'RMS':
                decay_val = decay
                min_method = tf.train.RMSPropOptimizer(learning_rate=sample, decay=decay)
            else:
                raise ValueError()

            min_iter = self.__check_optimizer(min_method.minimize(f[0]), f, st_p, pre_generated_functions)

            with tf.Session() as sess:
                thewriter_t.writerow({'method': method, 'learning_rate': sample, 'decay': decay_val, 'value': sess.run(f[0], feed_dict={pre_generated_functions.x: st_p}), "step_size": 0})
            # Write to file for every learning rate
            for i in range(len(min_iter[1])):
                thewriter_t.writerow({'method': method, 'learning_rate': sample, 'decay': decay_val, 'value': min_iter[1][i], "step_size": min_iter[4][i]})

            if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
                res = min_iter
                sample_lr = sample
        return sample_lr, res[0], res[2], res[3]

    # This is the essentially the most important function. All of the writing to files happens here
    def process(self):
        # Create test directory if not exist
        if not os.path.exists('output/' + self.path + '/'):
            os.makedirs('output/' + self.path + '/')
        # Disable logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Opening files to write general information for tests and functions
        with open('output/' + self.path + '/' + 'results' + '.csv', 'w', newline='') as output, open('output/' + self.path + '/' + 'summary.txt', 'w', newline='') as output_f:
            # Column names for tests files (i.e. c1, c2)
            fieldnames = ['number', 'start', 'optimal_lr', 'iterations', 'method']
            # Writing columns to file. Look up python's csv module
            thewriter = csv.DictWriter(output, fieldnames=fieldnames)
            thewriter.writeheader()

            dims_amount = len(self.function_amount.keys())
            k = 0
            # For each dimension in the given dictionary do the following
            for dim in self.function_amount.keys():
                # Create a generator object for functions
                pre_generated_functions = generator.FunGen(dim, self.function_amount[dim], self.path)
                generated_functions = pre_generated_functions.generate(self.proto_function)
                f_n = self.function_amount[dim]
                # Create a required amount of functions of a certain dimension
                for i in range(f_n):
                    # Generate a function
                    f = generated_functions[i]
                    # Generate starting points
                    starting_points = generator.generate_points(dim, self.m, self.r, f[2])
                    # Counter for points (used to name files)
                    l = 0

                    # For a file containing info about functions write down the function number
                    output_f.write('Function ' + str(i+1) + '/' + str(f_n) + ' of dimension ' + str(dim) + '\n')
                    # Write the generated function to function file
                    output_f.write(str(f[1]) + '\n\n')
                    output_f.write('Expected minimum coordinates: \t')
                    output_f.write(str(f[2]) + '\n')
                    output_f.write('Expected minimum value: \t')
                    output_f.write(str(f[3]) + '\n')

                    # Write all the generated points to function file
                    output_f.write('\nStarting points:\n')
                    d = 1
                    # For each starting point do the following
                    for st_p in starting_points:
                        output_f.write('-' * 25 + '\n')
                        output_f.write(str(d) + ': \t' + str(st_p) + '\n\n')
                        # Open file to write down trends
                        with open('output/' + self.path + '/' + 'test_' + str(dim) + '_' + str(i) + '_' + str(l) + '.csv', 'w', newline='') as output_t:
                            # Column names for test files (i.e. trend_*_*)
                            fieldnames_t = ['method', 'learning_rate', 'decay', 'value', 'step_size']
                            # Writing columns to file. Look up python's csv module
                            thewriter_t = csv.DictWriter(output_t, fieldnames=fieldnames_t)
                            thewriter_t.writeheader()

                            # For each method to be tested do the following
                            for method in self.methods.keys():
                                # Calculate optimal learning rate
                                res = self.__optimal_learning_rate(method, f, self.methods[method], st_p, pre_generated_functions, thewriter_t)

                                output_f.write(method.upper() + ': \t')
                                output_f.write(str(res[2]))
                                output_f.write(';\t Value: ')
                                output_f.write(str(res[3]) + ' in ' + str(res[1]) + ' iterations with ' + str(res[0]) + ' learning rate' + '\n')
                                # Write results of the calculations above to the tests file
                                thewriter.writerow({'number': i, 'start': st_p, 'optimal_lr': res[0], 'iterations': res[1], 'method': method})
                        output_f.write('-' * 25 + '\n')
                        l += 1
                        d += 1
                    output_f.write('\n\n')
                    print('Completed', str(i+1) + '/' + str(self.function_amount[dim]), 'of', dim, 'dimension',)
                print('--- Completed', dim, 'dimension (total:', str(k+1) + '/' + str(dims_amount) + ')\n')
                k += 1
        print('All completed')


# def main():
#     # Initial approximations
#     n_funcs = ind.n_funcs
#     n_points = ind.n_points
#     r = ind.r
#     methods = ind.methods
#
#     # Start the process
#     process(loss_functions.sum_powers, n_funcs, n_points, r, methods)
#
#
# if __name__ == "__main__":
#     main()
