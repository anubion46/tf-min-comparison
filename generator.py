import tensorflow as tf
from random import random, seed
import numpy as np


class FunGen:
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        self.functions = []
        self.x = tf.get_variable('x', [dim])

    def generate(self, function_type):
        seed(10)
        if function_type == 'c1':
            power = 4/3
        elif function_type == 'c2':
            power = 2
        else:
            raise ValueError()

        for j in range(self.n):
            a_random = [random() for _ in range(self.dim)]
            min_point_random = [random() for _ in range(self.dim)]
            min_value_random = random()

            a = tf.constant(a_random)
            min_point = tf.constant(min_point_random)
            min_value = tf.constant(min_value_random)

            f = tf.add(min_value, tf.reduce_sum(tf.multiply(a, tf.pow(tf.subtract(self.x, min_point), power))))
            self.functions.append((f, min_point_random, min_value_random, str(self.dim) + '_' + str(j), a_random, power))
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
