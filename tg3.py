import loss_functions as lf
import method_run as mr
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


m = 1
iterations = 300


# train_losses = {
#                 'sum_sin': [lf.sum_sin, {40: 2, 15: 2}],
#                 'sum_powers': [lf.sum_powers, {40: 2, 15: 2}],
#                 'rastrigin': [lf.rastrigin, {40: 2, 15: 2}],
#                 'deb01': [lf.deb01, {40: 2, 15: 2}],
#                 'alpine01': [lf.alpine01, {40: 2, 15: 2}]
#                 }

# test_losses = {
#                'sum_sin': [lf.sum_sin, {40: 4, 15: 4}],
#                'sum_powers': [lf.sum_powers, {40: 4, 15: 4}],
#                'rastrigin': [lf.rastrigin, {40: 4, 15: 4}],
#                'deb01': [lf.deb01, {40: 4, 15: 4}],
#                'alpine01': [lf.alpine01, {4: 4, 15: 4}]
#                }

train_losses = {'alpine01': [lf.alpine01, {15: 1}]}

test_losses = {'alpine01': [lf.alpine01, {15: 1}]}


# mr.runGradientDescent(train_losses, test_losses, m, iterations, np.geomspace(0.001, 10.0, 5))
#
# mr.runAdam(train_losses, test_losses, m, iterations, np.geomspace(0.001, 0.1, 5))
#
# mr.runMomentum(train_losses, test_losses, m, iterations, np.geomspace(0.001, 0.1, 5))

mr.runAdagrad(train_losses, test_losses, m, iterations, np.geomspace(0.0001, 1.0, 15))

# mr.runAdadelta(train_losses, test_losses, m, iterations, np.geomspace(0.01, 100, 5), np.geomspace(0.9, 0.9999, 2))
#
# mr.runRMS(train_losses, test_losses, m, iterations, np.geomspace(0.01, 10.0, 5), np.geomspace(0.9, 0.9999, 2))

