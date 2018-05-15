import loss_functions
import test_template
import numpy as np


def main():
    # Initial approximations
    test = test_template.Test(path='test10',
                              proto_function=loss_functions.csendes,
                              function_amount={4: 2},
                              m=2, r=0.5,
                              methods={'momentum': np.geomspace(.01, .3, 4),
                                       'adam': np.geomspace(.4, 1.0, 4),
                                       'adadelta': np.geomspace(0.1, 10, 4),
                                       'adagrad': np.geomspace(0.1, 10, 4),
                                       'RMS': [0.5, 0.75, 1, 1.25],
                                       },
                              iter_threshold=1500,
                              decay=[0.9, 0.99999],
                              eps=1e-6)

    test.process()


if __name__ == "__main__":
    main()
