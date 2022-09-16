#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import ClassVar

import numpy as np


@dataclass
class Point:
    x: float = 0
    y: float = 0

    def __sub__(self, other: Point):
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __add__(self, other: Point):
        return Point(x=self.x + other.x, y=self.y + other.y)

    def manhattan_distance(self, point: Point = None):
        return abs(self.x) + abs(self.x) if point is None else abs(self.x - point.x) + abs(self.x - point.y)

    def sq_distance(self, point: Point = None):
        return self.x ** 2 + self.x ** 2 if point is None else (self.x - point.x) ** 2 + (self.x - point.y) ** 2


@dataclass
class Star:
    star_id: int = field(init=False)
    pos_history: list[Point] = field(default_factory=list)
    detection_history: list[int] = field(default_factory=list)
    detection_confidence: float = 0
    blinking_freq: float = 0
    speed_vec: Point = field(default_factory=Point)
    lifetime: float = 10.

    default_lifetime: ClassVar[float] = 10.
    next_star_id: ClassVar[int] = 0

    def __post_init__(self):
        self.star_id = Star.next_star_id
        Star.next_star_id += 1

    @property
    def position(self):
        return self.pos_history[-1] if self.pos_history else Point(0, 0)

    def expected_pos(self):
        return self.pos_history[-1] + self.speed_vec

    def remove_old_history(self, max_history_length=20):
        self.pos_history = self.pos_history[-max_history_length:]
        self.detection_history = self.detection_history[-max_history_length:]

    def add_detection(self, video_fps: float, desired_blinking_freq: float, pos: Point = None, freq_threshold=3.):
        if pos is None:
            self.detection_history.append(0)
            self.lifetime -= 1
        else:
            self.detection_history.append(1)
            self.pos_history.append(pos)
            self.lifetime = self.default_lifetime

        self.remove_old_history()
        self.blinking_freq = video_fps * sum(self.detection_history) / len(self.detection_history)
        if abs(self.blinking_freq - desired_blinking_freq) < freq_threshold:
            self.detection_confidence += 1
        else:
            self.detection_confidence -= 2


def time_it(func):
    """ Function to add a timer to the execution of a given function. """

    def wrapper(*args, **kwargs):
        """ Returned function with the time tracking added. """

        start_time = perf_counter()
        data = func(*args, **kwargs)
        end_time = perf_counter()

        print(f"  {func.__name__} took {end_time - start_time:.10f} seconds",
              end="\n")
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
