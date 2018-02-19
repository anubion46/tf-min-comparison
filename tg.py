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
        res_iter = 0
        for _ in range(ind.n_iter):
            if len(fs) == ind.th:
                trend.append(np.median(fs))
                del fs[0:ind.th//2]
            sess.run(optimizer)
            fs.append(sess.run(f[0]))
            n += 1
            if check_min(fs[-1], f[2]):
                res_iter = n
    return res_iter, trend


def run_momentum(method, f, sample_lrs, st_p, funcs, thewriter):
    res, sample_lr = [-1, -1], -1
    for sample in sample_lrs:
        min_iter = check_optimizer(tf.train.MomentumOptimizer(learning_rate=sample, momentum=0.99, use_nesterov=True).minimize(f[0]), f, st_p, funcs)

        # Write to file for every learning rate
        for i in range(len(min_iter[1])):
            thewriter.writerow({'METHOD': method, 'LR': sample, 'VALUE': min_iter[1][i]})

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
            thewriter.writerow({'METHOD': method, 'LR': sample, 'VALUE': min_iter[1][i]})

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
            thewriter.writerow({'METHOD': method, 'LR': sample, 'VALUE': min_iter[1][i]})

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
            thewriter_t.writerow({'METHOD': method, 'LR': sample, 'VALUE': min_iter[1][i]})

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


def processC1(n_funcs, n_points, r):
    for dim in n_funcs.keys():
        with open('output/c1.csv', 'w', newline='') as output:
            fieldnames = ['№F', 'START', 'OLR', 'ITER', 'METHOD']
            thewriter = csv.DictWriter(output, fieldnames=fieldnames)
            thewriter.writeheader()
            momentum, adam, adadelta, adagrad = [], [], [], []
            for i in range(n_funcs[dim]):
                funcs = generator.FunGen(dim, n_funcs[dim])
                f = funcs.c1()[i]
                starting_points = generator.PointGen(dim, n_points, r, f[1]).create_points()
                for st_p in starting_points:
                    momentum.append(optimal_learning_rate('momentum', f, ind.momentum_lr, st_p, funcs))
                    adam.append(optimal_learning_rate('adam', f,  ind.adam_lr, st_p, funcs))
                    adadelta.append(optimal_learning_rate('adadelta', f, ind.adadelta_lr, st_p, funcs))
                    adagrad.append(optimal_learning_rate('adagrad', f, ind.adagrad_lr, st_p, funcs))
                print('\n')

            k = 1
            for i in range(n_funcs[dim]):
                for j in range(n_points):
                    thewriter.writerow(
                        {'№F': i + 1, 'START': momentum[i*k + j][2], 'OLR': momentum[i*k + j][0], 'ITER': momentum[i*k + j][1], 'METHOD': 'MOMENTUM'})
                    thewriter.writerow(
                        {'№F': i + 1, 'START': adam[i*k + j][2], 'OLR': adam[i*k + j][0], 'ITER': adam[i*k + j][1], 'METHOD': 'ADAM'})
                    thewriter.writerow(
                        {'№F': i + 1, 'START': adadelta[i*k + j][2], 'OLR': adadelta[i][0], 'ITER': adadelta[i*k + j][1], 'METHOD': 'ADADELTA'})
                    thewriter.writerow(
                        {'№F': i + 1, 'START': adagrad[i*k + j][2], 'OLR': adagrad[i][0], 'ITER': adagrad[i*k + j][1], 'METHOD': 'ADAGRAD'})
                k += 1


def processC2(n_funcs, n_points, r):
    with open('output/c2.csv', 'w', newline='') as output:
        fieldnames = ['№F', 'START', 'OLR', 'ITER', 'METHOD']
        thewriter = csv.DictWriter(output, fieldnames=fieldnames)
        thewriter.writeheader()
        for dim in n_funcs.keys():
            funcs = generator.FunGen(dim, n_funcs[dim])
            for i in range(n_funcs[dim]):
                f = funcs.c2()[i]
                starting_points = generator.PointGen(dim, n_points, r, f[1]).create_points()
                l = 0
                for st_p in starting_points:
                    with open('output/trend_' + f[3] + '_' + str(l) + '.csv', 'w', newline='') as output_t:
                        fieldnames_t = ['METHOD', 'LR', 'VALUE']
                        thewriter_t = csv.DictWriter(output_t, fieldnames=fieldnames_t)
                        thewriter_t.writeheader()
                        momentum_temp = optimal_learning_rate('momentum', f, ind.momentum_lr, st_p, funcs, thewriter_t)
                        adam_temp = optimal_learning_rate('adam', f,  ind.adam_lr, st_p, funcs, thewriter_t)
                        adadelta_temp = optimal_learning_rate('adadelta', f, ind.adadelta_lr, st_p, funcs, thewriter_t)
                        adagrad_temp = optimal_learning_rate('adagrad', f, ind.adagrad_lr, st_p, funcs, thewriter_t)

                    thewriter.writerow({'№F': i, 'START': momentum_temp[2], 'OLR': momentum_temp[0], 'ITER': momentum_temp[1][0], 'METHOD': 'MOMENTUM'})
                    thewriter.writerow({'№F': i, 'START': adam_temp[2], 'OLR': adam_temp[0], 'ITER': adam_temp[1][0], 'METHOD': 'ADAM'})
                    thewriter.writerow({'№F': i, 'START': adadelta_temp[2], 'OLR': adadelta_temp[0], 'ITER': adadelta_temp[1][0], 'METHOD': 'ADADELTA'})
                    thewriter.writerow({'№F': i, 'START': adagrad_temp[2], 'OLR': adagrad_temp[0], 'ITER': adagrad_temp[1][0], 'METHOD': 'ADAGRAD'})
                    l += 1


def processC1_mc():
    pass


def processC2_mc():
    pass


def main():
    # Initial approximations
    n_funcs = ind.n_funcs
    n_points = ind.n_points
    r = ind.r

    processC2(n_funcs, n_points, r)


if __name__ == "__main__":
    main()
