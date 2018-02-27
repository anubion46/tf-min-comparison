import tensorflow as tf
from random import random, seed
import numpy as np


def sum_func(x, min_point, min_value, a, power):
    res = min_value
    for i in range(len(min_point)):
        res += a[i]*(x[i] - min_point[i]) ** power
    return res


class FunGen:
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        self.functions = []
        self.x = []
        for i in range(self.dim):
            self.x.append(tf.Variable(dtype=tf.float32, initial_value=1.0, name=('x' + str(i))))

    def generate(self, function_type, mc=False):
        seed(10)
        if function_type == 'c1':
            power = 4/3
        elif function_type == 'c2':
            power = 2
        else:
            raise ValueError()

        if mc:
            for _ in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                a = [random() / (random() + 0.1) for _ in range(self.dim)]
                min_value = random()
                self.functions.append((sum_func, min_point, min_value, a, power))
            return self.functions

        else:
            for j in range(self.n):
                min_point = [random() for _ in range(self.dim)]
                a = [random() / (random() + 0.1) for _ in range(self.dim)]
                min_value = random()
                f = min_value
                for i in range(self.dim):
                    f += a[i] * (self.x[i] - min_point[i]) ** power
                self.functions.append((f, min_point, min_value, str(j) + '_' + str(self.dim), a, power))
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
