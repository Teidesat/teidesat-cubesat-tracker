import numpy as np


class SVector3D:
    def __init__(self, phi=0, theta=0, r=1):
        self.phi = phi
        self.theta = theta
        self.r = r

    def to_cartesian(self, phi_inc=0, theta_inc=0):
        theta = self.theta + theta_inc
        phi = self.phi + phi_inc
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(theta)
        return CVector3D(x, y, z)

    def __repr__(self):
        return '(ϕ: {:.3f}, θ: {:.3f}, r: {:.3f})'.format(self.phi, self.theta, self.r)

    def __add__(self, other):
        return SVector3D(self.phi + other.phi, self.theta + other.theta, self.r)

    def __sub__(self, other):
        return SVector3D(self.phi - other.phi, self.theta - other.theta, self.r)


class CVector3D:
    def __init__(self, x=0, y=0, z=0):
        self.coord = np.array((x, y, z))

    def __repr__(self):
        return '(x: {:.3f}, y: {:.3f}, z: {:.3f})'.format(self.x, self.y, self.z)

    def to_spherical(self):
        r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        phi = np.arctan(self.y / self.x)
        theta = np.arccos(self.z / r)
        return SVector3D(phi, theta, r)

    @property
    def x(self): return self.coord[0]

    @property
    def y(self): return self.coord[1]

    @property
    def z(self): return self.coord[2]

    @x.setter
    def x(self, value): self.coord[0] = value

    @y.setter
    def y(self, value): self.coord[1] = value

    @z.setter
    def z(self, value): self.coord[2] = value

    def calc_angle(self, u):
        return np.arccos(self.calc_cos_angle(u))

    def calc_cos_angle(self, u):
        if isinstance(u, SVector3D):
            u = u.to_cartesian()
        return np.dot(self.coord, u.coord) / (np.linalg.norm(u.coord) * np.linalg.norm(self.coord))

    def normalize(self):
        coords = self.coord / np.linalg.norm(self.coord)
        return CVector3D(*coords)

    def distance_to_line(self, p1, p2):
        return np.linalg.norm(np.cross(p1.coord - p2.coord, p2.coord - self.coord)) \
               / np.linalg.norm(p1.coord - p2.coord)

    def distance_to_point(self, point):
        return np.linalg.norm(self.coord - point.coord)

    def cross(self, other):
        if isinstance(other, SVector3D):
            other = other.to_cartesian()
        return CVector3D(*(np.cross(self.coord, other.coord)))

    def __add__(self, other):
        return CVector3D(*(self.coord + other.coord))

    def __sub__(self, other):
        return CVector3D(*(self.coord - other.coord))

    def __mul__(self, other):
        return CVector3D(*(self.coord * other))

    def __getitem__(self, pos):
        return self.coord[pos]
