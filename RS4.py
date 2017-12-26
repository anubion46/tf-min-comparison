import tensorflow as tf
import generator
import numpy as np
import csv


def check_min(val, m):
    return np.abs(val - m) <= eps


def check_optimizer(optimizer, f, st_p):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # reset values to wrong
        for j in range(len(st_p)):
            funcs.x[j].assign(st_p[j]).eval()
        n = 0
        fs = []
        while True:
            sess.run(optimizer)
            fs.append(sess.run(f[0]))
            n += 1
            if n >= n_iter:
                return -1
            if check_min(fs[len(fs) - 1], f[2]):
                return n


def optimal_learning_rate(method, f, samples, st_p):
    res = -1
    sample_lr = -1
    for sample in samples:
        if method == 'momentum':
            min_iter = check_optimizer(tf.train.MomentumOptimizer(learning_rate=sample, momentum=0.99, use_nesterov=True).minimize(f[0]), f, st_p)
        elif method == 'adam':
            min_iter = check_optimizer(tf.train.AdamOptimizer(learning_rate=sample, epsilon=eps).minimize(f[0]), f, st_p)
        elif method == 'adadelta':
            min_iter = check_optimizer(tf.train.AdadeltaOptimizer(learning_rate=sample, epsilon=eps).minimize(f[0]), f, st_p)
        elif method == 'adagrad':
            min_iter = check_optimizer(tf.train.AdagradOptimizer(learning_rate=sample).minimize(f[0]), f, st_p)
        else:
            raise ValueError()
        if res < 0 or (res > min_iter != -1):
            res = min_iter
            sample_lr = sample
    print(method, sample_lr, res, st_p)
    return [sample_lr, res, st_p]


# Initial approximations
n_iter = 500
n_funcs = 3
eps = 1e-6
dim = 2
n_points = 3
r = 0.7


with open('output/c2.csv', 'w', newline='') as output:
    fieldnames = ['№F', 'START', 'OLR', 'ITER', 'METHOD']
    thewriter = csv.DictWriter(output, fieldnames=fieldnames)
    thewriter.writeheader()
    momentum, adam, adadelta, adagrad = [], [], [], []
    for i in range(n_funcs):
        funcs = generator.FunGen(dim, n_funcs)
        f = funcs.c2()[i]
        starting_points = generator.PointGen(dim, n_points, r, f[1]).create_points()
        for st_p in starting_points:
            momentum.append(optimal_learning_rate('momentum', f, np.linspace(.5, 1.0, 4), st_p))
            adam.append(optimal_learning_rate('adam', f, np.linspace(0.01, 10, 4), st_p))
            adadelta.append(optimal_learning_rate('adadelta', f, np.linspace(10, 3000, 4), st_p))
            adagrad.append(optimal_learning_rate('adagrad', f, np.linspace(0.1, 25, 4), st_p))
        print('\n')

    for i in range(n_funcs):
        for j in range(n_points):
            thewriter.writerow(
                {'№F': i + 1, 'START': momentum[i][2], 'OLR': momentum[i][0], 'ITER': momentum[i][1], 'METHOD': 'MOMENTUM'})
            thewriter.writerow(
                {'№F': i + 1, 'START': adam[i][2], 'OLR': adam[i][0], 'ITER': adam[i][1], 'METHOD': 'ADAM'})
            thewriter.writerow(
                {'№F': i + 1, 'START': adadelta[i][2], 'OLR': adadelta[i][0], 'ITER': adadelta[i][1], 'METHOD': 'ADADELTA'})
            thewriter.writerow(
                {'№F': i + 1, 'START': adagrad[i][2], 'OLR': adagrad[i][0], 'ITER': adagrad[i][1], 'METHOD': 'ADAGRAD'})

