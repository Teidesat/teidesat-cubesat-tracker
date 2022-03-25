#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

from time import time

import numpy as np


def time_it(func):
    """ Docstring """  # ToDo: redact docstring

    def wrapper(*args, **kwargs):
        """ Docstring """  # ToDo: redact docstring

        start_time = time()
        data = func(*args, **kwargs)
        end_time = time()

        print(f"{func.__name__} took {end_time - start_time} seconds")
        return data

    return wrapper


class Distance:
    """ Class docstring """  # ToDo: redact docstring

    @staticmethod
    def euclidean(point_a, point_b):
        """ Docstring """  # ToDo: redact docstring

        x_dist = point_a[0] - point_b[0]
        y_dist = point_a[1] - point_b[1]
        return np.sqrt(x_dist * x_dist + y_dist * y_dist)

    @staticmethod
    def manhattan(point_a, point_b):
        """ Docstring """  # ToDo: redact docstring

        x_dist = point_a[0] - point_b[0]
        y_dist = point_a[1] - point_b[1]
        return abs(x_dist) + abs(y_dist)

    @staticmethod
    def between_spherical(point_a, point_b, radius=1):
        """ Docstring """  # ToDo: redact docstring

        # theta_from, theta_to, phi_from, phi_to = a[0], b[0], a[1], b[1]
        x_dist = abs(
            np.sin(point_a[0]) * np.cos(point_a[1]) -
            np.sin(point_b[0]) * np.cos(point_b[1]))
        y_dist = abs(
            np.sin(point_a[0]) * np.sin(point_a[1]) -
            np.sin(point_b[0]) * np.sin(point_b[1]))
        z_dist = abs(np.cos(point_a[0]) - np.cos(point_b[0]))
        return np.sqrt(
            (x_dist * x_dist + y_dist * y_dist + z_dist * z_dist) * radius)


def find_neighbours(item, items, radius, dist_func):
    """ Docstring """  # ToDo: redact docstring

    neighbours = []
    distances = []
    for neighbour in items:
        if item != neighbour:
            distance = dist_func(item, neighbour)
            if distance < radius:
                neighbours.append(neighbour)
                distances.append(distance)
    return neighbours, distances


def find_angle_vectors(center, first, second):
    """ Docstring """  # ToDo: redact docstring

    x0 = first[0] - center[0]
    y0 = first[1] - center[1]
    x1 = second[0] - center[0]
    y1 = second[1] - center[1]
    mag = np.sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1))
    return np.arccos(np.dot([x0, y0], [x1, y1]) / mag)
