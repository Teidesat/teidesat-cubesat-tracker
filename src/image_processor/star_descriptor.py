from collections import defaultdict

import numpy as np

from src.utils import time_it


class StarDescriptor:
    def __init__(self, star, first_ref, second_ref, first_dist, second_dist, angle):
        self.star = star
        self.first_ref = first_ref
        self.second_ref = second_ref
        self.rel_dist = min([first_dist, second_dist]) / max([first_dist, second_dist])
        self.angle = angle

        if angle < 0:
            print('Warning, angle < 0 in descriptor')

    def __str__(self):
        star = '({:7.2f}, {:7.2f})'.format(*self.star)
        first_ref = '({:7.2f}, {:7.2f})'.format(*self.first_ref)
        second_ref = '({:7.2f}, {:7.2f})'.format(*self.second_ref)
        return '{} -> {} {} rel_dist = {:.4f} angle = {:.4f}'.format(star, first_ref, second_ref, self.rel_dist, self.angle)

    @staticmethod
    @time_it
    def build_descriptors(stars, px_radius, dist_func):
        descriptors = []
        for star in stars:
            neighbours, distance = StarDescriptor.find_neighbours(star, stars, px_radius, dist_func)

            for i in range(len(neighbours)):
                for j in range(i + 1, len(neighbours)):
                    angle = StarDescriptor.find_angle(star, neighbours[i], neighbours[j])
                    descriptors.append(StarDescriptor(star=star,
                                                      first_ref=neighbours[i],
                                                      second_ref=neighbours[j],
                                                      first_dist=distance[i],
                                                      second_dist=distance[j],
                                                      angle=angle))
        return descriptors

    @staticmethod
    def find_angle(center, first, second):
        x0 = first[0] - center[0]
        y0 = first[1] - center[1]
        x1 = second[0] - center[0]
        y1 = second[1] - center[1]
        mag = np.sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1))
        return np.arccos(np.dot([x0, y0], [x1, y1]) / mag)

    @staticmethod
    def find_neighbours(item, items, radius, dist_func):
        neighbours = []
        distances = []
        for neighbour in items:
            if item != neighbour:
                distance = dist_func(item, neighbour)
                if distance < radius:
                    neighbours.append(neighbour)
                    distances.append(distance)
        return neighbours, distances
