import loss_functions as lf
import method_runner as mr
import numpy as np
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


m = 2
iterations = 30


# train_losses = {
#                 'sum_sin': [lf.sum_sin, {3: 2, 5: 2, 7: 2, 10: 2, 15: 2, 20: 2}],
#                 'sum_powers': [lf.sum_powers, {3: 2, 5: 2, 7: 2, 10: 2, 15: 2, 20: 2}],
#                 'rastrigin': [lf.rastrigin, {3: 2, 5: 2, 7: 2, 10: 2, 15: 2, 20: 2}],
#                 'deb01': [lf.deb01, {3: 2, 5: 2, 7: 2, 10: 2, 15: 2, 20: 2}],
#                 'alpine01': [lf.alpine01, {3: 2, 5: 2, 7: 2, 10: 2, 15: 2, 20: 2}]
#                 }
#
# test_losses = {
#                'sum_sin': [lf.sum_sin, {3: 4, 5: 4, 7: 4, 10: 4, 15: 4, 20: 4, 25: 4}],
#                'sum_powers': [lf.sum_powers, {3: 4, 5: 4, 7: 4, 10: 4, 15: 4, 20: 4, 25: 4}],
#                'rastrigin': [lf.rastrigin, {3: 4, 5: 4, 7: 4, 10: 4, 15: 4, 20: 4, 25: 4}],
#                'deb01': [lf.deb01, {3: 4, 5: 4, 7: 4, 10: 4, 15: 4, 20: 4, 25: 4}],
#                'alpine01': [lf.alpine01, {3: 4, 5: 4, 7: 4, 10: 4, 15: 4, 20: 4, 25: 4}]
#                }

train_losses = {
                'sum_sin': [lf.sum_sin, {3: 2, 5: 2}],
                'sum_powers': [lf.sum_powers, {3: 2, 5: 2}],
                'rastrigin': [lf.rastrigin, {3: 2, 5: 2}],
                'deb01': [lf.deb01, {3: 2, 5: 2}],
                'alpine01': [lf.alpine01, {3: 2, 5: 2}]
                }

test_losses = {
               'sum_sin': [lf.sum_sin, {3: 4, 5: 4}],
               'sum_powers': [lf.sum_powers, {3: 4, 5: 4}],
               'rastrigin': [lf.rastrigin, {3: 4, 5: 4}],
               'deb01': [lf.deb01, {3: 4, 5: 4}],
               'alpine01': [lf.alpine01, {3: 4, 5: 4}]
               }


dest = 'output/adaptive2/'
if not os.path.exists(dest):
    os.mkdir(dest)
runner = mr.MethodRunner(iterations, m, train_losses, test_losses, save=dest)

nspace = 10
runner.runGradientDescent(np.geomspace(0.001, 10.0, nspace))
runner.runAdam(np.geomspace(0.001, 0.1, nspace))
runner.runMomentum(np.geomspace(0.001, 0.1, nspace))
runner.runAdagrad(np.geomspace(0.0001, 10.0, nspace))
runner.runAdadelta(np.geomspace(0.01, 10, nspace), np.geomspace(0.9, 0.9999, 2))
runner.runRMS(np.geomspace(0.01, 10.0, nspace), np.geomspace(0.9, 0.9999, 2))

runner.show_plot()
