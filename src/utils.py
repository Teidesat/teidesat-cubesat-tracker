from time import time
import numpy as np


def time_it(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        data = func(*args, **kwargs)
        t1 = time()
        print('{:15s} took {:.5f} seconds'.format(func.__name__, t1 - t0))
        return data
    return wrapper


class Distance:
    @staticmethod
    def euclidean(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return np.sqrt(dx * dx + dy * dy)

    @staticmethod
    def manhattan(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return abs(dx) + abs(dy)

    @staticmethod
    def between_spherical(a, b, r=1):
        # theta_from, theta_to, phi_from, phi_to = a[0], b[0], a[1], b[1]
        dx = abs(np.sin(a[0]) * np.cos(a[1]) - np.sin(b[0]) * np.cos(b[1]))
        dy = abs(np.sin(a[0]) * np.sin(a[1]) - np.sin(b[0]) * np.sin(b[1]))
        dz = abs(np.cos(a[0]) - np.cos(b[0]))
        return np.sqrt((dx * dx + dy * dy + dz * dz) * r)
