import method_trainer as mtrain
import method_tester as mtest
import numpy as np
import matplotlib.pyplot as plt
import json
import csv


def search_learning_rate(data, metric, search_decay=False):
    best_learning_rate = list(data.keys())[0]
    if not search_decay:
        best_val = data[best_learning_rate][metric]
        for learning_rate in data.keys():
            if data[learning_rate][metric] <= best_val:
                best_val = data[learning_rate][metric]
                best_learning_rate = learning_rate
        return best_learning_rate
    else:
        best_decay = list(data[best_learning_rate].keys())[0]
        best_val = data[best_learning_rate][best_decay][metric]

        for learning_rate in data.keys():
            for decay in data[learning_rate].keys():
                if data[learning_rate][decay][metric] < best_val:
                    best_val = data[learning_rate][decay][metric]
                    best_decay = decay
                    best_learning_rate = learning_rate
        return best_learning_rate, best_decay


class MethodRunner:
    def __init__(self, iterations, m, train_losses, test_losses, save=''):
        self.iterations = iterations
        self.m = m
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.save = save
        self.x = [i for i in range(self.iterations)]
        if save:
            with open(save + 'test_results.csv', 'w') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                thewriter.writeheader()

    def show_plot(self):
        plt.show()

    def runGradientDescent(self, learning_rates):
        gradient_descent_learning_rates = learning_rates
        training_gradient_descent = mtrain.training_gradient_descent(self.train_losses, self.m, self.iterations, gradient_descent_learning_rates)

        # print('Gradient Descent results:')
        # print(json.dumps(training_gradient_descent, indent=4), '\n')

        with open(self.save + 'gd_test_results.json', 'w') as file:
            json.dump(training_gradient_descent, file)

        print('Gradient training done')

        best_gd_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_gd_params.append(search_learning_rate(training_gradient_descent, metric))
        # print('Best gradient descent learning rates:', " ".join([str(i) for i in best_gd_params]), '\n')

        test_gd_best = mtest.test_gradient_descent(self.test_losses, self.m, self.iterations, best_gd_params[0], np.amin)
        test_gd_worst = mtest.test_gradient_descent(self.test_losses, self.m, self.iterations, best_gd_params[1], np.amax)
        test_gd_mean = mtest.test_gradient_descent(self.test_losses, self.m, self.iterations, best_gd_params[2], np.mean)
        test_gd_median = mtest.test_gradient_descent(self.test_losses, self.m, self.iterations, best_gd_params[3], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_gd_best, color='red', alpha=0.5, label='Best, LR:' + str(round(best_gd_params[0], 5)))
        plt.plot(self.x, test_gd_median, color='purple', alpha=0.5, label='Median, LR:' + str(round(best_gd_params[1], 5)))
        plt.plot(self.x, test_gd_worst, color='green', alpha=0.5, label='Worst, LR:' + str(round(best_gd_params[2], 5)))
        plt.plot(self.x, test_gd_mean, color='blue', alpha=0.5, label='Mean, LR:' + str(round(best_gd_params[3], 5)))

        plt.title('Gradient descent')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'gd_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'best', 'iteration': i, 'value': test_gd_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'worst', 'iteration': i, 'value': test_gd_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'mean', 'iteration': i, 'value': test_gd_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'gradient_descent', 'metric': 'median', 'iteration': i, 'value': test_gd_median[i]})

        print('Gradient testing done\n')

    def runAdam(self, learning_rates):
        adam_learning_rates = learning_rates
        training_adam = mtrain.training_adam(self.train_losses, self.m, self.iterations, adam_learning_rates)
        # print('Adam results:')
        # print(json.dumps(training_adam, indent=4), '\n')

        with open(self.save + 'adam_test_results.json', 'w') as file:
            json.dump(training_adam, file)

        print('Adam training done')

        best_adam_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_adam_params.append(search_learning_rate(training_adam, metric))
        # print('Best adam learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

        test_adam_best = mtest.test_adam(self.test_losses, self.m, self.iterations, best_adam_params[0], np.amin)
        test_adam_worst = mtest.test_adam(self.test_losses, self.m, self.iterations, best_adam_params[1], np.amax)
        test_adam_mean = mtest.test_adam(self.test_losses, self.m, self.iterations, best_adam_params[2], np.mean)
        test_adam_median = mtest.test_adam(self.test_losses, self.m, self.iterations, best_adam_params[3], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_adam_best, color='red', alpha=0.5, label='Best, LR: ' + str(round(best_adam_params[0], 5)))
        plt.plot(self.x, test_adam_worst, color='green', alpha=0.5, label='Worst, LR: ' + str(round(best_adam_params[1], 5)))
        plt.plot(self.x, test_adam_mean, color='blue', alpha=0.5, label='Mean, LR: ' + str(round(best_adam_params[2], 5)))
        plt.plot(self.x, test_adam_median, color='purple', alpha=0.5, label='Median, LR: ' + str(round(best_adam_params[3], 5)))
        plt.title('Adam')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adam_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'best', 'iteration': i, 'value': test_adam_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'worst', 'iteration': i, 'value': test_adam_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'mean', 'iteration': i, 'value': test_adam_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adam', 'metric': 'median', 'iteration': i, 'value': test_adam_median[i]})

        print('Adam testing done\n')

    def runMomentum(self, learning_rates):
        momentum_learning_rates = learning_rates
        training_momentum = mtrain.training_momentum(self.train_losses, self.m, self.iterations, momentum_learning_rates)
        # print('Momentum results:')
        # print(json.dumps(training_momentum, indent=4), '\n')

        with open(self.save + 'momentum_test_results.json', 'w') as file:
            json.dump(training_momentum, file)

        print('Momentum training done')

        best_momentum_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_momentum_params.append(search_learning_rate(training_momentum, metric))
        # print('Best momentum learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

        test_momentum_best = mtest.test_momentum(self.test_losses, self.m, self.iterations, best_momentum_params[0], np.amin)
        test_momentum_worst = mtest.test_momentum(self.test_losses, self.m, self.iterations, best_momentum_params[1], np.amax)
        test_momentum_mean = mtest.test_momentum(self.test_losses, self.m, self.iterations, best_momentum_params[2], np.mean)
        test_momentum_median = mtest.test_momentum(self.test_losses, self.m, self.iterations, best_momentum_params[3], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_momentum_best, color='red', alpha=0.5, label='Best, LR: ' + str(round(best_momentum_params[0], 5)))
        plt.plot(self.x, test_momentum_worst, color='green', alpha=0.5, label='Worst, LR: ' + str(round(best_momentum_params[1], 5)))
        plt.plot(self.x, test_momentum_mean, color='blue', alpha=0.5, label='Mean, LR: ' + str(round(best_momentum_params[2], 5)))
        plt.plot(self.x, test_momentum_median, color='purple', alpha=0.5, label='Median, LR: ' + str(round(best_momentum_params[3], 5)))
        plt.title("Nesterov's Momentum")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'momentum_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'best', 'iteration': i, 'value': test_momentum_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'worst', 'iteration': i, 'value': test_momentum_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'mean', 'iteration': i, 'value': test_momentum_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'momentum', 'metric': 'median', 'iteration': i, 'value': test_momentum_median[i]})

        print('Momentum testing done\n')

    def runAdagrad(self, learning_rates):
        adagrad_learning_rates = learning_rates
        training_adagrad = mtrain.training_adam(self.train_losses, self.m, self.iterations, adagrad_learning_rates)
        # print('Adagrad results:')
        # print(json.dumps(training_adagrad, indent=4), '\n')

        with open(self.save + 'adagrad_test_results.json', 'w') as file:
            json.dump(training_adagrad, file)

        print('Adagrad training done')

        best_adagrad_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_adagrad_params.append(search_learning_rate(training_adagrad, metric))
        # print('Best adam learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

        test_adagrad_best = mtest.test_adagrad(self.test_losses, self.m, self.iterations, best_adagrad_params[0], np.amin)
        test_adagrad_worst = mtest.test_adagrad(self.test_losses, self.m, self.iterations, best_adagrad_params[1], np.amax)
        test_adagrad_mean = mtest.test_adagrad(self.test_losses, self.m, self.iterations, best_adagrad_params[2], np.mean)
        test_adagrad_median = mtest.test_adagrad(self.test_losses, self.m, self.iterations, best_adagrad_params[3], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_adagrad_best, color='red', alpha=0.5, label='Best, LR: ' + str(round(best_adagrad_params[0], 5)))
        plt.plot(self.x, test_adagrad_worst, color='green', alpha=0.5, label='Worst, LR: ' + str(round(best_adagrad_params[1], 5)))
        plt.plot(self.x, test_adagrad_mean, color='blue', alpha=0.5, label='Mean, LR: ' + str(round(best_adagrad_params[2], 5)))
        plt.plot(self.x, test_adagrad_median, color='purple', alpha=0.5, label='Median, LR: ' + str(round(best_adagrad_params[3], 5)))
        plt.title("Adagrad")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adagrad_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'best', 'iteration': i, 'value': test_adagrad_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'worst', 'iteration': i, 'value': test_adagrad_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'mean', 'iteration': i, 'value': test_adagrad_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adagrad', 'metric': 'median', 'iteration': i, 'value': test_adagrad_median[i]})

        print('Adagrad testing done\n')

    def runAdadelta(self, learning_rates, decays):
        adadelta_learning_rates = learning_rates
        adadelta_decays = decays
        training_adadelta = mtrain.training_adadelta(self.train_losses, self.m, self.iterations, adadelta_learning_rates, adadelta_decays)
        # print('Adadelta results:')
        # print(json.dumps(training_adadelta, indent=4), '\n')

        with open(self.save + 'adadelta_test_results.json', 'w') as file:
            json.dump(training_adadelta, file)

        print('Adadelta training done')

        best_adadelta_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_adadelta_params.append(search_learning_rate(training_adadelta, metric, search_decay=True))
        # print('Best adadelta learning rates and decays:', " ".join([str(i) for i in best_gd_params]), '\n')

        test_adadelta_best = mtest.test_adadelta(self.test_losses, self.m, self.iterations, best_adadelta_params[0][0], best_adadelta_params[0][1], np.amin)
        test_adadelta_worst = mtest.test_adadelta(self.test_losses, self.m, self.iterations, best_adadelta_params[1][0], best_adadelta_params[1][1], np.amax)
        test_adadelta_mean = mtest.test_adadelta(self.test_losses, self.m, self.iterations, best_adadelta_params[2][0], best_adadelta_params[2][1], np.mean)
        test_adadelta_median = mtest.test_adadelta(self.test_losses, self.m, self.iterations, best_adadelta_params[3][0], best_adadelta_params[3][1], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_adadelta_best, color='red', alpha=0.5, label='Best, LR: ' + str(round(best_adadelta_params[0][0], 5)) + '; DEC: ' + str(round(best_adadelta_params[0][1], 5)))
        plt.plot(self.x, test_adadelta_worst, color='green', alpha=0.5, label='Worst, LR: ' + str(round(best_adadelta_params[1][0], 5)) + '; DEC: ' + str(round(best_adadelta_params[1][1], 5)))
        plt.plot(self.x, test_adadelta_mean, color='blue', alpha=0.5, label='Mean, LR: ' + str(round(best_adadelta_params[2][0], 5)) + '; DEC: ' + str(round(best_adadelta_params[2][1], 5)))
        plt.plot(self.x, test_adadelta_median, color='purple', alpha=0.5, label='Median, LR: ' + str(round(best_adadelta_params[3][0], 5)) + '; DEC: ' + str(round(best_adadelta_params[3][1], 5)))
        plt.title("Adadelta")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'adadelta_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'best', 'iteration': i, 'value': test_adadelta_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'worst', 'iteration': i, 'value': test_adadelta_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'mean', 'iteration': i, 'value': test_adadelta_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'adadelta', 'metric': 'median', 'iteration': i, 'value': test_adadelta_median[i]})

        print('Adadelta testing done\n')

    def runRMS(self, learning_rates, decays):
        rms_learning_rates = learning_rates
        rms_decays = decays
        training_rms = mtrain.training_rms(self.train_losses, self.m, self.iterations, rms_learning_rates, rms_decays)
        # print('RMSProb results:')
        # print(json.dumps(training_rms, indent=4), '\n')

        with open(self.save + 'rms_test_results.json', 'w') as file:
            json.dump(training_rms, file)

        print('RMS training done')

        best_rms_params = []
        for metric in ['best', 'worst', 'mean', 'median']:
            best_rms_params.append(search_learning_rate(training_rms, metric, search_decay=True))
        # print('Best RMSProb learning rates and decays:', " ".join([str(i) for i in best_gd_params]), '\n')

        test_rms_best = mtest.test_rms(self.test_losses, self.m, self.iterations, best_rms_params[0][0], best_rms_params[0][1], np.amin)
        test_rms_worst = mtest.test_rms(self.test_losses, self.m, self.iterations, best_rms_params[1][0], best_rms_params[1][1], np.amax)
        test_rms_mean = mtest.test_rms(self.test_losses, self.m, self.iterations, best_rms_params[2][0], best_rms_params[2][1], np.mean)
        test_rms_median = mtest.test_rms(self.test_losses, self.m, self.iterations, best_rms_params[3][0], best_rms_params[3][1], np.median)

        fig = plt.figure()
        plt.plot(self.x, test_rms_best, color='red', alpha=0.5, label='Best, LR: ' + str(round(best_rms_params[0][0], 5)) + '; DEC: ' + str(round(best_rms_params[0][1], 5)))
        plt.plot(self.x, test_rms_median, color='purple', alpha=0.5, label='Median, LR: ' + str(round(best_rms_params[1][0], 5)) + '; DEC: ' + str(round(best_rms_params[1][1], 5)))
        plt.plot(self.x, test_rms_worst, color='green', alpha=0.5, label='Worst, LR: ' + str(round(best_rms_params[2][0], 5)) + '; DEC: ' + str(round(best_rms_params[2][1], 5)))
        plt.plot(self.x, test_rms_mean, color='blue', alpha=0.5, label='Mean, LR: ' + str(round(best_rms_params[3][0], 5)) + '; DEC: ' + str(round(best_rms_params[3][1], 5)))

        plt.title("RMSProb")
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.ylim([0.0, 1.2])
        plt.grid(True)
        plt.legend()
        fig.savefig(self.save + 'rms_train.png', dpi=fig.dpi)
        if self.save:
            with open(self.save + 'test_results.csv', 'a') as file:
                fieldnames = ['method', 'metric', 'iteration', 'value']
                thewriter = csv.DictWriter(file, fieldnames=fieldnames)
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'best', 'iteration': i, 'value': test_rms_best[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'worst', 'iteration': i, 'value': test_rms_worst[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'mean', 'iteration': i, 'value': test_rms_mean[i]})
                for i in range(self.iterations):
                    thewriter.writerow({'method': 'rms', 'metric': 'median', 'iteration': i, 'value': test_rms_median[i]})

        print('RMS testing done\n')
