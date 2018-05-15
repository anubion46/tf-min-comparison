import method_trainer as mtrain
import method_tester as mtest
import numpy as np
import matplotlib.pyplot as plt
import json


def search_learning_rate(data, metric, search_decay=False):
    best_val = 1
    best_learning_rate = 0
    if not search_decay:
        for learning_rate in data.keys():
            if data[learning_rate][metric] < best_val:
                best_val = data[learning_rate][metric]
                best_learning_rate = learning_rate
        return best_learning_rate
    else:
        best_decay = 0
        for learning_rate in data.keys():
            for decay in data[learning_rate].keys():
                if data[learning_rate][decay][metric] < best_val:
                    best_val = data[learning_rate][decay][metric]
                    best_decay = decay
                    best_learning_rate = learning_rate
        return best_learning_rate, best_decay


def runGradientDescent(train_losses, test_losses, m, iterations, learning_rates):
    gradient_descent_learning_rates = learning_rates
    training_gradient_descent = mtrain.training_gradient_descent(train_losses, m, iterations, gradient_descent_learning_rates)
    # print('Gradient Descent results:')
    # print(json.dumps(training_gradient_descent, indent=4), '\n')

    with open('output/gd_test_results.json', 'w') as file:
        json.dump(training_gradient_descent, file)

    best_gd_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_gd_params.append(search_learning_rate(training_gradient_descent, metric))
    # print('Best gradient descent learning rates:', " ".join([str(i) for i in best_gd_params]), '\n')

    test_gd_best = mtest.test_gradient_descent(test_losses, m, iterations, best_gd_params[0], min)
    test_gd_worst = mtest.test_gradient_descent(test_losses, m, iterations, best_gd_params[1], max)
    test_gd_mean = mtest.test_gradient_descent(test_losses, m, iterations, best_gd_params[2], np.mean)
    test_gd_median = mtest.test_gradient_descent(test_losses, m, iterations, best_gd_params[3], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_gd_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_gd_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_gd_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_gd_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/gd_train.png', dpi=fig.dpi)


def runAdam(train_losses, test_losses, m, iterations, learning_rates):
    adam_learning_rates = learning_rates
    training_adam = mtrain.training_adam(train_losses, m, iterations, adam_learning_rates)
    # print('Adam results:')
    # print(json.dumps(training_adam, indent=4), '\n')

    with open('output/adam_test_results.json', 'w') as file:
        json.dump(training_adam, file)

    best_adam_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_adam_params.append(search_learning_rate(training_adam, metric))
    # print('Best adam learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

    test_adam_best = mtest.test_adam(test_losses, m, iterations, best_adam_params[0], min)
    test_adam_worst = mtest.test_adam(test_losses, m, iterations, best_adam_params[1], max)
    test_adam_mean = mtest.test_adam(test_losses, m, iterations, best_adam_params[2], np.mean)
    test_adam_median = mtest.test_adam(test_losses, m, iterations, best_adam_params[3], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_adam_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adam_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adam_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adam_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/adam_train.png', dpi=fig.dpi)


def runMomentum(train_losses, test_losses, m, iterations, learning_rates):
    momentum_learning_rates = learning_rates
    training_momentum = mtrain.training_momentum(train_losses, m, iterations, momentum_learning_rates)
    # print('Momentum results:')
    # print(json.dumps(training_momentum, indent=4), '\n')

    with open('output/momentum_test_results.json', 'w') as file:
        json.dump(training_momentum, file)

    best_momentum_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_momentum_params.append(search_learning_rate(training_momentum, metric))
    # print('Best momentum learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

    test_momentum_best = mtest.test_momentum(test_losses, m, iterations, best_momentum_params[0], min)
    test_momentum_worst = mtest.test_momentum(test_losses, m, iterations, best_momentum_params[1], max)
    test_momentum_mean = mtest.test_momentum(test_losses, m, iterations, best_momentum_params[2], np.mean)
    test_momentum_median = mtest.test_momentum(test_losses, m, iterations, best_momentum_params[3], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_momentum_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_momentum_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_momentum_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_momentum_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/momentum_train.png', dpi=fig.dpi)


def runAdagrad(train_losses, test_losses, m, iterations, learning_rates):
    adagrad_learning_rates = learning_rates
    training_adagrad = mtrain.training_adam(train_losses, m, iterations, adagrad_learning_rates)
    # print('Adagrad results:')
    # print(json.dumps(training_adagrad, indent=4), '\n')

    with open('output/adagrad_test_results.json', 'w') as file:
        json.dump(training_adagrad, file)

    best_adagrad_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_adagrad_params.append(search_learning_rate(training_adagrad, metric))
    # print('Best adam learning rates:', " ".join([str(i) for i in best_adam_params]), '\n')

    test_adagrad_best = mtest.test_adagrad(test_losses, m, iterations, best_adagrad_params[0], min)
    test_adagrad_worst = mtest.test_adagrad(test_losses, m, iterations, best_adagrad_params[1], max)
    test_adagrad_mean = mtest.test_adagrad(test_losses, m, iterations, best_adagrad_params[2], np.mean)
    test_adagrad_median = mtest.test_adagrad(test_losses, m, iterations, best_adagrad_params[3], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_adagrad_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adagrad_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adagrad_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adagrad_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/adagrad_train.png', dpi=fig.dpi)


def runAdadelta(train_losses, test_losses, m, iterations, learning_rates, decays):
    adadelta_learning_rates = learning_rates
    adadelta_decays = decays
    training_adadelta = mtrain.training_adadelta(train_losses, m, iterations, adadelta_learning_rates, adadelta_decays)
    # print('Adadelta results:')
    # print(json.dumps(training_adadelta, indent=4), '\n')

    with open('output/adadelta_test_results.json', 'w') as file:
        json.dump(training_adadelta, file)

    best_adadelta_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_adadelta_params.append(search_learning_rate(training_adadelta, metric, search_decay=True))
    # print('Best adadelta learning rates and decays:', " ".join([str(i) for i in best_gd_params]), '\n')

    test_adadelta_best = mtest.test_adadelta(test_losses, m, iterations, best_adadelta_params[0][0], best_adadelta_params[0][1], min)
    test_adadelta_worst = mtest.test_adadelta(test_losses, m, iterations, best_adadelta_params[1][0], best_adadelta_params[1][1], max)
    test_adadelta_mean = mtest.test_adadelta(test_losses, m, iterations, best_adadelta_params[2][0], best_adadelta_params[2][1], np.mean)
    test_adadelta_median = mtest.test_adadelta(test_losses, m, iterations, best_adadelta_params[3][0], best_adadelta_params[3][1], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_adadelta_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adadelta_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adadelta_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_adadelta_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/adadelta_train.png', dpi=fig.dpi)


def runRMS(train_losses, test_losses, m, iterations, learning_rates, decays):
    rms_learning_rates = learning_rates
    rms_decays = decays
    training_rms = mtrain.training_rms(train_losses, m, iterations, rms_learning_rates, rms_decays)
    # print('RMSProb results:')
    # print(json.dumps(training_rms, indent=4), '\n')

    with open('output/rms_test_results.json', 'w') as file:
        json.dump(training_rms, file)

    best_rms_params = []
    for metric in ['best', 'worst', 'mean', 'median']:
        best_rms_params.append(search_learning_rate(training_rms, metric, search_decay=True))
    # print('Best RMSProb learning rates and decays:', " ".join([str(i) for i in best_gd_params]), '\n')

    test_rms_best = mtest.test_rms(test_losses, m, iterations, best_rms_params[0][0], best_rms_params[0][1], min)
    test_rms_worst = mtest.test_rms(test_losses, m, iterations, best_rms_params[1][0], best_rms_params[1][1], max)
    test_rms_mean = mtest.test_rms(test_losses, m, iterations, best_rms_params[2][0], best_rms_params[2][1], np.mean)
    test_rms_median = mtest.test_rms(test_losses, m, iterations, best_rms_params[3][0], best_rms_params[3][1], np.median)

    fig = plt.figure()
    plt.plot([i for i in range(iterations)], test_rms_best, color='red', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_rms_worst, color='green', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_rms_mean, color='blue', alpha=0.5)
    plt.plot([i for i in range(iterations)], test_rms_median, color='purple', alpha=0.5)
    # plt.show()
    fig.savefig('output/rms_train.png', dpi=fig.dpi)
