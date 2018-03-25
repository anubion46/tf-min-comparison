import loss_functions
import test_template
import numpy as np


def main():
    # Initial approximations
    test2 = test_template.Test(path='test2/',
                               proto_function=loss_functions.sum_powers,
                               function_amount={20: 1},
                               m=2, r=0.9,
                               methods={'momentum': np.geomspace(.01, .3, 15),
                                        'adam': np.geomspace(.4, 1.0, 15),
                                        'adadelta': np.geomspace(0.1, 10, 15),
                                        'adagrad': np.geomspace(0.1, 10, 15)},
                               iter_threshold=1000,
                               eps=1e-5,
                               window=20)

    test2.process()

if __name__ == "__main__":
    main()
