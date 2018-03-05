"""
This is a setup file containing:
- n_functions (dictionary): number of dimensions: number of functions
- n_iter: upper threshold of iterations
- eps: error of calculations
- methods: methods to be tested
- n_points: number of starting points
- r: radius for hypersphere
- window: window for trends
and learning rates for each method
"""

import numpy as np

# ~ 2 hours for this data

# Dictionary containing number functions for each present dimension
n_funcs = {3: 2}

# Methods to be tested
methods = ['momentum', 'adam', 'adadelta', 'adagrad']

# Upper threshold for the number of iterations and epsilon
n_iter = 3000
eps = 1e-7

# Window for trends
window = 60

# Input values for the number of points to be generated and radius for hypersphere
n_points = 2
r = 0.9

# Learning rates to be tested during calculations
momentum_lr = np.geomspace(.01, .3, 5)
adam_lr = np.geomspace(.4, 1.0, 5)
adadelta_lr = np.geomspace(0.1, 10, 5)
adagrad_lr = np.geomspace(0.1, 10, 5)
