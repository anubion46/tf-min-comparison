import numpy as np
from random import random, seed
import generator


def monte_carlo_local(func, dim, amount):
    seed(10)
    points = [[random() for _ in range(dim)] for _ in range(amount)]
    temp = [func[0](min_point=func[1], min_value=func[2], a=func[3], power=func[4], x=point) for point in points]
    min_index = np.argmin(temp)
    return temp[min_index], points[min_index]


def monte_carlo(functions, dim, amount):
    return [monte_carlo_local(f, dim, amount) for f in functions]


amount = 15000
dim = 6
functions = generator.FunGen(dim, 1).generate('c2', mc=True)
print(monte_carlo(functions, dim, amount))
print(functions[0][2])
