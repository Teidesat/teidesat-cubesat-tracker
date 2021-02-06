import cv2
import numpy as np


class SphereProjection:
    def __init__(self):
        self.width = None
        self.height = None
        self.img = None

    def from_spherical(self, phi_arr, theta_arr):
        raise NotImplementedError

    def from_spherical_int(self, phi_arr, theta_arr):
        x, y = np.round(self.from_spherical(phi_arr, theta_arr)).astype('int32')
        return x, y

    def project(self, phi_arr, theta_arr, items, draw_function):
        raise NotImplementedError

    def project_grid(self, color, thickness=1, v_lines=24, v_points=10, h_lines=12, h_points=10):
        quantity = np.complex(1j)
        vertical = np.mgrid[0:2*np.pi:(v_lines*quantity), 0:np.pi:(v_points*quantity)].reshape(2, -1).T
        horizontal = np.mgrid[0:np.pi:(h_lines*quantity), 0:2*np.pi:(h_points*quantity)].reshape(2, -1).T

        points = self.from_spherical_int(vertical[:, 0], vertical[:, 1])
        points = np.array(points).swapaxes(0, 1)  # Transform [[0,1,2,3...],[4,5,6,...]] into [[0,4],[1,5],...]
        points = points.reshape(v_lines, -1, 2)  # Divide coordinates in groups so we have h_lines groups
        cv2.drawContours(self.img, points, -1, color, thickness=thickness, lineType=cv2.LINE_AA)

        points = self.from_spherical_int(horizontal[:, 1], horizontal[:, 0])
        points = np.array(points).swapaxes(0, 1)
        points = points.reshape(h_lines, -1, 2)
        cv2.drawContours(self.img, points, -1, color, thickness=thickness, lineType=cv2.LINE_AA)

    def fill_circle(self, point, radius, color):
        cv2.circle(self.img, point, int(radius), color, thickness=-1, lineType=cv2.LINE_AA)


class SphereEquirectProjection(SphereProjection):
    """ If height is none, height will be adjusted automatically """

    def __init__(self, width, height=None):
        super().__init__()
        self.width = width
        self.height = height if height else width // 2
        self.img = np.zeros((self.height, self.width, 3), dtype='uint8')

    def from_spherical(self, phi, theta):
        """ Project theta and phi angles as x,y coordinates """
        x = self.width * phi / (2 * np.pi)
        y = self.height * theta / np.pi
        return x, y

    def project(self, phi_arr, theta_arr, items, draw_function):
        x_arr, y_arr = self.from_spherical_int(phi_arr, theta_arr)
        for x, y, item in zip(x_arr, y_arr, items):
            draw_function(x, y, item)
