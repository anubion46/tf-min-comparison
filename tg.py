import tensorflow as tf
import generator
import numpy as np
import csv
import initial_data as ind


def check_min(val, m):
    return np.abs(val - m) <= ind.eps


def check_optimizer(optimizer, f, st_p, funcs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # reset values to wrong
        for j in range(len(st_p)):
            funcs.x[j].assign(st_p[j]).eval()
        n = 0
        fs = []
        trend = []
        res_iter = ind.n_iter
        for _ in range(ind.n_iter):
            if len(fs) == ind.th:
                trend.append(np.median(fs))
                del fs[0:ind.th//2]
            sess.run(optimizer)
            fs.append(sess.run(f[0]))
            n += 1
            if check_min(fs[-1], f[2]) and res_iter > n:
                res_iter = n
    return res_iter, trend


def run_momentum(method, f, sample_lrs, st_p, funcs, thewriter):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        min_iter = check_optimizer(tf.train.MomentumOptimizer(learning_rate=sample, momentum=0.99, use_nesterov=True).minimize(f[0]), f, st_p, funcs)

        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter.writerow({'method': method, 'learning_rate': sample, 'value': min_iter[1][i]})

        if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
            res = min_iter
            sample_lr = sample
    return sample_lr, res, st_p


def run_adam(method, f, sample_lrs, st_p, funcs, thewriter):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        min_iter = check_optimizer(tf.train.AdamOptimizer(learning_rate=sample, epsilon=ind.eps).minimize(f[0]), f, st_p, funcs)

        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter.writerow({'method': method, 'learning_rate': sample, 'value': min_iter[1][i]})

        if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
            res = min_iter
            sample_lr = sample
    return sample_lr, res, st_p


def run_adadelta(method, f, sample_lrs, st_p, funcs, thewriter):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        min_iter = check_optimizer(tf.train.AdadeltaOptimizer(learning_rate=sample, epsilon=ind.eps).minimize(f[0]), f, st_p, funcs)

        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter.writerow({'method': method, 'learning_rate': sample, 'value': min_iter[1][i]})

        if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
            res = min_iter
            sample_lr = sample
    return sample_lr, res, st_p


def run_adagrad(method, f, sample_lrs, st_p, funcs, thewriter_t):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        min_iter = check_optimizer(tf.train.AdagradOptimizer(learning_rate=sample).minimize(f[0]), f, st_p, funcs)

        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter_t.writerow({'method': method, 'learning_rate': sample, 'value': min_iter[1][i]})

        if (res[0] < 0 and min_iter[0] != -1) or (res[0] > min_iter[0] != -1):
            res = min_iter
            sample_lr = sample
    return sample_lr, res, st_p


# Identify optimal learning rate of a method for a function
def optimal_learning_rate(method, f, sample_lrs, st_p, funcs, thewriter_t):
    if method == 'momentum':
        result = run_momentum(method, f, sample_lrs, st_p, funcs, thewriter_t)
    elif method == 'adam':
        result = run_adam(method, f, sample_lrs, st_p, funcs, thewriter_t)
    elif method == 'adadelta':
        result = run_adadelta(method, f, sample_lrs, st_p, funcs, thewriter_t)
    elif method == 'adagrad':
        result = run_adagrad(method, f, sample_lrs, st_p, funcs, thewriter_t)
    else:
        raise ValueError()
    print(result)
    return result


def process(function_type, n_funcs, n_points, r, methods):
    with open('output/' + function_type + '.csv', 'w', newline='') as output, open('output/functions', 'w', newline='') as output_f:
        fieldnames = ['number', 'start', 'optimal_lr', 'iterations', 'method']
        thewriter = csv.DictWriter(output, fieldnames=fieldnames)
        thewriter.writeheader()

        for dim in n_funcs.keys():
            funcs = generator.FunGen(dim, n_funcs[dim])
            for i in range(n_funcs[dim]):
                f = funcs.generate(function_type)[i]
                starting_points = generator.PointGen(dim, n_points, r, f[1]).create_points()
                l = 0

                output_f.write('-'*5 + 'Function ' + str(l+1) + '-'*5 + '\n\n')
                for h in range(dim):
                    output_f.write(str(f[4][h]) + '(x' + str(h + 1) + ' + ' + str(f[1][h]) + ')^' + str(f[5]) + ' + ')
                output_f.write(str(f[2]) + '\n\n\n')
                output_f.write('Starting points:\n')
                for d in range(n_points):
                    output_f.write(str(d + 1) + ': ' + str(starting_points[d]) + '\n')
                output_f.write('\n' + '-'*10 + '\n')

                for st_p in starting_points:
                    with open('output/trend_' + f[3] + '_' + str(l) + '.csv', 'w', newline='') as output_t:
                        fieldnames_t = ['method', 'learning_rate', 'value']
                        thewriter_t = csv.DictWriter(output_t, fieldnames=fieldnames_t)
                        thewriter_t.writeheader()
                        for met in methods:
                            res = optimal_learning_rate(met, f, ind.momentum_lr, st_p, funcs, thewriter_t)
                            thewriter.writerow({'number': i, 'start': res[2], 'optimal_lr': res[0], 'iterations': res[1][0], 'method': met})
                    l += 1


def main():
    # Initial approximations
    n_funcs = ind.n_funcs
    n_points = ind.n_points
    r = ind.r
    methods = ind.methods

    process('c2', n_funcs, n_points, r, methods)


if __name__ == "__main__":
    main()
