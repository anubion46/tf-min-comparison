import loss_functions
import test_template
import numpy as np


def main():
    # Initial approximations
    test1 = test_template.Test(path='test7',
                               proto_function=loss_functions.damavandi,
                               function_amount={3: 1},
                               m=2, r=5,
                               methods={'adadelta': [0.5, 0.75, 1],
                                        'RMS': [0.5, 0.75, 1]},
                               iter_threshold=500,
                               window=10,
                               rho=0.99999,
                               eps=1e-6)

    test2 = test_template.Test(path='test8',
                               proto_function=loss_functions.damavandi,
                               function_amount={3: 1},
                               m=2, r=5,
                               methods={'adadelta': [0.5, 0.75, 1],
                                        'RMS': [0.5, 0.75, 1]},
                               iter_threshold=500,
                               window=10,
                               rho=0.95,
                               eps=1e-6)
    test1.process()
    test2.process()

if __name__ == "__main__":
    main()
