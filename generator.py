import tensorflow as tf
from random import random, seed
import numpy as np


class FunGen:
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        self.functions = []
        self.x = []
        for i in range(self.dim):
            self.x.append(tf.Variable(dtype=tf.float32, initial_value=1.0, name=('x' + str(i))))

    def c2(self):
        seed(10)
        for _ in range(self.n):
            min_point = [random() for _ in range(self.dim)]
            min_value = random()
            f = min_value
            for i in range(self.dim):
                a = random() / (random() + 0.1)
                f += a * (self.x[i] - min_point[i]) ** 2

            self.functions.append((f, min_point, min_value))
        return self.functions

    def c1(self):
        seed(10)
        for _ in range(self.n):
            min_point = [random() for _ in range(self.dim)]
            min_value = random()
            f = min_value
            for i in range(self.dim):
                a = random() / (random() + 0.1)
                f += a * (self.x[i] - min_point[i]) ** (4/3)

            self.functions.append((f, min_point, min_value))
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
