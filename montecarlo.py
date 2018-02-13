import numpy as np
from random import random, seed
import generator


def monte_carlo_local(func, point, dim, r, amount):
    seed(10)
    points = []
    point = np.array(point)
    points.append(point)
    for _ in range(amount):
        curr_point = point
        for j in range(dim):
            curr_point[j] = point[j] + r * (random() - 0.5)
            if curr_point[j] > 1:
                curr_point[j] = 1
            if curr_point[j] < 0:
                curr_point[j] = 0
            points.append(curr_point)

    minimum = func[0](x=point, min_point=func[1], min_value=func[2])
    min_point = points[0]
    for i in points:
        min1 = func[0](x=i, min_point=func[1], min_value=func[2])
        if min1 < minimum:
            minimum = min1
            min_point = i

    return minimum, min_point


def monte_carlo(functions, points, dim, amount, r):
    for i in functions:
        minimum = [i[0](x=np.array(points[0]), min_point=i[1], min_value=i[2]), points[0]]
        for j in points:
            min_cur = monte_carlo_local(i, j, dim, r, amount)
            if min_cur[0] < minimum[0]:
                minimum = min_cur

    return minimum


amount = 100
dim = 3
ls = 3
r = (dim / (ls ** 2)) ** dim / dim
points = generator.f(dim, ls)
functions = generator.FunGen(dim, 1).c2(mc=True)
print(monte_carlo(functions, points, dim, amount, r))
print(functions[0][2])
