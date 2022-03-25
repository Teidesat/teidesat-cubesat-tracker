#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import cv2 as cv
import numpy as np


class SphereProjection:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self):
        self.width = None
        self.height = None
        self.img = None

    # ToDo: "from_spherical" function not implemented
    def from_spherical(self, phi_arr, theta_arr):
        """ Docstring """  # ToDo: redact docstring
        raise NotImplementedError

    def from_spherical_int(self, phi_arr, theta_arr):
        """ Docstring """  # ToDo: redact docstring

        x_arr, y_arr = np.round(self.from_spherical(phi_arr,
                                                    theta_arr)).astype("int32")
        return x_arr, y_arr

    # ToDo: "project" function not implemented
    def project(self, phi_arr, theta_arr, items, draw_function):
        """ Docstring """  # ToDo: redact docstring
        raise NotImplementedError

    def project_grid(self,
                     color,
                     thickness=1,
                     v_lines=24,
                     v_points=10,
                     h_lines=12,
                     h_points=10):
        """ Docstring """  # ToDo: redact docstring

        quantity = np.complex(1j)
        vertical = np.mgrid[0:2 * np.pi:(v_lines * quantity),
                            0:np.pi:(v_points * quantity)].reshape(2, -1).T
        horizontal = np.mgrid[0:np.pi:(h_lines * quantity),
                              0:2 * np.pi:(h_points * quantity)].reshape(
                                  2, -1).T

        points = self.from_spherical_int(vertical[:, 0], vertical[:, 1])
        # Transform [[0,1,2,3...],[4,5,6,...]] into [[0,4],[1,5],...]
        points = np.array(points).swapaxes(0, 1)
        # Divide coordinates in groups so we have h_lines groups
        points = points.reshape(v_lines, -1, 2)
        cv.drawContours(self.img,
                        points,
                        -1,
                        color,
                        thickness=thickness,
                        lineType=cv.LINE_AA)

        points = self.from_spherical_int(horizontal[:, 1], horizontal[:, 0])
        points = np.array(points).swapaxes(0, 1)
        points = points.reshape(h_lines, -1, 2)
        cv.drawContours(self.img,
                        points,
                        -1,
                        color,
                        thickness=thickness,
                        lineType=cv.LINE_AA)

    def fill_circle(self, point, radius, color):
        """ Docstring """  # ToDo: redact docstring

        cv.circle(self.img,
                  point,
                  int(radius),
                  color,
                  thickness=-1,
                  lineType=cv.LINE_AA)


class SphereEquirectProjection(SphereProjection):
    """ Class docstring """  # ToDo: redact docstring

    # If height is none, height will be adjusted automatically
    def __init__(self, width, height=None):
        super().__init__()
        self.width = width
        self.height = height if height else width // 2
        self.img = np.zeros((self.height, self.width, 3), dtype="uint8")

    def from_spherical(self, phi_arr, theta_arr):
        """ Project theta and phi angles as x,y coordinates """

        x_arr = self.width * phi_arr / (2 * np.pi)
        y_arr = self.height * theta_arr / np.pi
        return x_arr, y_arr

    def project(self, phi_arr, theta_arr, items, draw_function):
        """ Docstring """  # ToDo: redact docstring

        x_arr, y_arr = self.from_spherical_int(phi_arr, theta_arr)
        for x_arr, y_arr, item in zip(x_arr, y_arr, items):
            draw_function(x_arr, y_arr, item)
