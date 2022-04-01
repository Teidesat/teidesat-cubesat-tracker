#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

from src.utils import find_neighbours, find_angle_vectors
# from src.image_processor.star_descriptor import StarDescriptor


class Descriptor:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self, first_ref, second_ref, first_dist, second_dist, angle):
        self.first_ref = first_ref
        self.second_ref = second_ref
        self.rel_dist = (min([first_dist, second_dist]) /
                         max([first_dist, second_dist]))
        self.angle = angle

        if angle < 0:
            print("Warning, angle < 0 in descriptor.")

    def __str__(self):
        return (f"[{self.first_ref} {self.second_ref} " +
                f"d: {self.rel_dist} a: {self.angle}]")


class Star:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self, cols, index):
        self.cols = cols
        self.index = index

    def __str__(self):
        name = self.cols["proper"][self.index]
        star_id = self.cols["id"][self.index]
        if not isinstance(name, str):
            name = f"Unnamed-{star_id}"
        return f"Star({name})"

    def __getitem__(self, item):
        return self.cols[item][self.index]

    def __setitem__(self, key, value):
        self.cols[key][self.index] = value

    def build_descriptor(self, star_group, px_radius, dist_func):
        """ Docstring """  # ToDo: redact docstring

        neighbours, distance = find_neighbours(self, star_group, px_radius,
                                               dist_func)

        for i, _ in enumerate(neighbours):
            for j in range(i + 1, len(neighbours)):
                angle = find_angle_vectors(self.get_pos(),
                                           neighbours[i].get_pos(),
                                           neighbours[j].get_pos())
                self["descriptor"] = Descriptor(first_ref=neighbours[i],
                                                second_ref=neighbours[j],
                                                first_dist=distance[i],
                                                second_dist=distance[j],
                                                angle=angle)

    # ToDo: "get_pos" function not implemented
    def get_pos(self):
        """ Docstring """  # ToDo: redact docstring
        raise NotImplementedError


class ImageStar:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self, x_coord, y_coord, ref=None):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.ref = ref  # Real Star in the database where it belongs to
