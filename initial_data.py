"""
This is a setup file containing:
- n_functions (dictionary): number of dimensions: number of functions
- n_iter: upper threshold of iterations
- eps: error of calculations
- n_points: number of starting points
- r: radius for hypersphere
and learning rates for each method
"""

import numpy as np


n_funcs = {3: 2, 10: 1}
methods = ['momentum', 'adam', 'adadelta', 'adagrad']

n_iter = 5000
eps = 1e-6

n_points = 3
r = 0.9
th = 200

momentum_lr = np.linspace(.01, .3, 6)
adam_lr = np.linspace(.4, 1.0, 6)
adadelta_lr = np.linspace(0.1, 12, 6)
adagrad_lr = np.linspace(0.1, 12, 6)
