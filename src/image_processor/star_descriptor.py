#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import numpy as np

from utils import time_it


class StarDescriptor:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(
        self,
        star,
        first_ref,
        second_ref,
        first_dist,
        second_dist,
        angle,
    ):
        self.star = star
        self.first_ref = first_ref
        self.second_ref = second_ref
        self.rel_dist = (min([first_dist, second_dist]) /
                         max([first_dist, second_dist]))
        self.angle = angle

        if angle < 0:
            print("Warning, angle < 0 in descriptor")

    def __str__(self):
        star = f"({self.star[0]}, {self.star[1]})"
        first_ref = f"({self.first_ref[0]}, {self.first_ref[1]})"
        second_ref = f"({self.second_ref[0]}, {self.second_ref[1]})"
        return (f"{star} -> {first_ref} {second_ref} " +
                f"rel_dist = {self.rel_dist} angle = {self.angle}")

    @staticmethod
    @time_it
    def build_descriptors(stars, px_radius, dist_func):
        """ Docstring """  # ToDo: redact docstring

        descriptors = []
        for star in stars:
            neighbours, distance = StarDescriptor.find_neighbours(
                star,
                stars,
                px_radius,
                dist_func,
            )

            for i, _ in enumerate(neighbours):
                for j in range(i + 1, len(neighbours)):
                    angle = StarDescriptor.find_angle(
                        star,
                        neighbours[i],
                        neighbours[j],
                    )
                    descriptors.append(
                        StarDescriptor(
                            star=star,
                            first_ref=neighbours[i],
                            second_ref=neighbours[j],
                            first_dist=distance[i],
                            second_dist=distance[j],
                            angle=angle,
                        ))
        return descriptors

    @staticmethod
    def find_angle(center, first, second):
        """ Docstring """  # ToDo: redact docstring

        x0 = first[0] - center[0]
        y0 = first[1] - center[1]
        x1 = second[0] - center[0]
        y1 = second[1] - center[1]
        mag = np.sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1))
        return np.arccos(np.dot([x0, y0], [x1, y1]) / mag)

    @staticmethod
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
