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


n_funcs = {3: 6, 5: 4, 10: 2}

n_iter = 10000
eps = 1e-6

n_points = 5
r = 0.9
th = 100

momentum_lr = np.linspace(.01, .3, 10)
adam_lr = np.linspace(.4, 1.0, 10)
adadelta_lr = np.linspace(0, 70, 10)
adagrad_lr = np.linspace(0, 70, 10)
