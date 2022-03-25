#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import numpy as np

from extra_math.vector3d import SVector3D


class Telescope:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self,
                 sky_target: SVector3D,
                 cam_alpha,
                 cam_beta,
                 width=1,
                 height=1):
        self.target = sky_target
        self.alpha = cam_alpha
        self.beta = cam_beta
        self.height = height
        self.width = width

    def to_screen2d(self, v: SVector3D):
        """ Docstring """  # ToDo: redact docstring

        dev = v - self.target
        if (not -self.alpha < dev.theta < self.alpha
                and not -self.beta < dev.phi < self.beta):
            return -1, -1

        half_w = np.sin(self.alpha)
        half_h = np.sin(self.beta)
        x_coord = (np.sin(dev.theta) - half_w) / (2 * half_w)
        y_coord = (np.sin(dev.phi) - half_h) / (2 * half_h)

        return -x_coord * self.width, -y_coord * self.height
