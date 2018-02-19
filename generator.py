import tensorflow as tf
from random import random, seed
import numpy as np
from itertools import tee, chain, product


def sum_squares(x, min_point, min_value):
    res = min_value
    for i in range(len(min_point)):
        res += (x[i] - min_point[i]) ** 2
    return res


class FunGen:
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        self.functions = []
        self.x = []
        for i in range(self.dim):
            self.x.append(tf.Variable(dtype=tf.float32, initial_value=1.0, name=('x' + str(i))))

    def c1(self, mc=False):
        seed(10)
        if mc:
            for _ in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                min_value = random()
                f = np.vectorize(sum_squares, excluded=['x', 'min_point', 'min_value'])

                self.functions.append((f, min_point, min_value))
            return self.functions

        else:
            for j in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                min_value = random()
                f = min_value
                for i in range(self.dim):
                    a = random() / (random() + 0.1)
                    f += a * (self.x[i] - min_point[i]) ** (4 / 3)

                self.functions.append((f, min_point, min_value, str(j) + '_' + str(self.dim)))
            return self.functions

    def c2(self, mc=False):
        seed(10)
        if mc:
            for _ in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                min_value = random()
                f = np.vectorize(sum_squares, excluded=['x', 'min_point', 'min_value'])
                self.functions.append((f, min_point, min_value))
            return self.functions

        else:
            for j in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                min_value = random()
                f = min_value
                for i in range(self.dim):
                    a = random() / (random() + 0.1)
                    f += a * (self.x[i] - min_point[i]) ** 2

                self.functions.append((f, min_point, min_value, str(j) + '_' + str(self.dim)))
            return self.functions


class PointGen:
    def __init__(self, dim, n, r, start):
        self.dim = dim
        self.n = n
        self.r = r
        self.points = []
        self.start = start

    def create_points(self):
        seed(10)
        for j in range(self.n):
            angles = [0]
            for i in range(1, self.dim):
                angles.append(random() * 2 * np.pi)
            point = []
            for k in range(self.dim):
                temp = self.r * np.cos(angles[k])
                for a in range(k+1, self.dim):
                    temp *= np.sin(angles[a])
                point.append(temp)
            self.points.append([x + y for x, y in zip(point, self.start)])
        return self.points


def pairwise(iterable):
    # s -> (s0,s1), (s1,s2), (s2, s3), ...
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def f(n, ls):
    values = np.linspace(0, 1.0, ls)
    return list(chain.from_iterable(product(x, repeat=n) for x in pairwise(values)))
