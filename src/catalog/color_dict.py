#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import colorsys

import numpy as np


class ColorDict:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self):
        self.colors = {}

    def __getitem__(self, key):
        try:
            return self.colors[key]
        except KeyError:
            color = self.random_color()
            self.colors[key] = color
            return color

    def rgb(self, key, brightness=1):
        """ Docstring """  # ToDo: redact docstring

        color = self[key]
        return colorsys.hsv_to_rgb(color[0], color[1], color[2] * brightness)

    def hsv(self, key, brightness=1):
        """ Docstring """  # ToDo: redact docstring

        color = self[key]
        return color[0], color[1], color[2] * brightness

    @staticmethod
    def random_color():
        """ Docstring """  # ToDo: redact docstring
        return np.random.uniform(0, 360), 1, 1
