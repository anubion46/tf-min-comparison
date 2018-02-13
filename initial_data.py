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


n_funcs = {3: 4}

n_iter = 300
eps = 1e-6

n_points = 4
r = 0.8

momentum_lr = np.linspace(.4, 1.0, 5)
adam_lr = np.linspace(.4, 1.0, 5)
adadelta_lr = np.linspace(10, 3000, 5)
adagrad_lr = np.linspace(0.1, 27, 5)
