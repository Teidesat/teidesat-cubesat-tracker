#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import numpy as np


class SVector3D:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self, phi=0, theta=0, r=1):
        self.phi = phi
        self.theta = theta
        self.r = r

    def to_cartesian(self, phi_inc=0, theta_inc=0):
        """ Docstring """  # ToDo: redact docstring

        theta = self.theta + theta_inc
        phi = self.phi + phi_inc
        x_coord = self.r * np.sin(theta) * np.cos(phi)
        y_coord = self.r * np.sin(theta) * np.sin(phi)
        z_coord = self.r * np.cos(theta)
        return CVector3D(x_coord, y_coord, z_coord)

    def __repr__(self):
        return f"(ϕ: {self.phi}, θ: {self.theta}, r: {self.r})"

    def __add__(self, other):
        return SVector3D(self.phi + other.phi, self.theta + other.theta,
                         self.r)

    def __sub__(self, other):
        return SVector3D(self.phi - other.phi, self.theta - other.theta,
                         self.r)


class CVector3D:
    """ Class docstring """  # ToDo: redact docstring

    def __init__(self, x_coord=0, y_coord=0, z_coord=0):
        self.coords = np.array((x_coord, y_coord, z_coord))

    def __repr__(self):
        return f"(x: {self.x_coord}, y: {self.y_coord}, z: {self.z_coord})"

    def to_spherical(self):
        """ Docstring """  # ToDo: redact docstring

        r = np.sqrt(self.x_coord**2 + self.y_coord**2 + self.z_coord**2)
        phi = np.arctan(self.y_coord / self.x_coord)
        theta = np.arccos(self.z_coord / r)

        return SVector3D(phi, theta, r)

    @property
    def x_coord(self):
        """ Docstring """  # ToDo: redact docstring
        return self.coords[0]

    @property
    def y_coord(self):
        """ Docstring """  # ToDo: redact docstring
        return self.coords[1]

    @property
    def z_coord(self):
        """ Docstring """  # ToDo: redact docstring
        return self.coords[2]

    @x_coord.setter
    def x_coord(self, value):
        self.coords[0] = value

    @y_coord.setter
    def y_coord(self, value):
        self.coords[1] = value

    @z_coord.setter
    def z_coord(self, value):
        self.coords[2] = value

    def calc_angle(self, other):
        """ Docstring """  # ToDo: redact docstring
        return np.arccos(self.calc_cos_angle(other))

    def calc_cos_angle(self, other):
        """ Docstring """  # ToDo: redact docstring

        if isinstance(other, SVector3D):
            other = other.to_cartesian()

        return (np.dot(self.coords, other.coords) /
                (np.linalg.norm(other.coords) * np.linalg.norm(self.coords)))

    def normalize(self):
        """ Docstring """  # ToDo: redact docstring

        coords = self.coords / np.linalg.norm(self.coords)
        return CVector3D(*coords)

    def distance_to_line(self, point_1, point_2):
        """ Docstring """  # ToDo: redact docstring

        return (np.linalg.norm(
            np.cross(point_1.coords - point_2.coords,
                     point_2.coords - self.coords)) /
                np.linalg.norm(point_1.coords - point_2.coords))

    def distance_to_point(self, point):
        """ Docstring """  # ToDo: redact docstring
        return np.linalg.norm(self.coords - point.coords)

    def cross(self, other):
        """ Docstring """  # ToDo: redact docstring

        if isinstance(other, SVector3D):
            other = other.to_cartesian()
        return CVector3D(*(np.cross(self.coords, other.coords)))

    def __add__(self, other):
        return CVector3D(*(self.coords + other.coords))

    def __sub__(self, other):
        return CVector3D(*(self.coords - other.coords))

    def __mul__(self, other):
        return CVector3D(*(self.coords * other))
        #! only "other"?
        #! isn't it "other.coords"?

    def __getitem__(self, pos):
        return self.coords[pos]
