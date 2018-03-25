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
n_funcs = {2: 2, 3: 3, 6: 2, 10: 2, 15: 2}

# Methods to be tested
# Learning rates to be tested during calculations
methods = {'momentum': np.geomspace(.01, .3, 10),
           'adam': np.geomspace(.4, 1.0, 10),
           'adadelta': np.geomspace(0.1, 10, 10),
           'adagrad': np.geomspace(0.1, 10, 10)}

# Upper threshold for the number of iterations and epsilon
n_iter = 5000
eps = 1e-5

# Window for trends
window = 30

# Input values for the number of points to be generated and radius for hypersphere
n_points = 5
r = 0.8
