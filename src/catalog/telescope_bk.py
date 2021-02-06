import matplotlib.pyplot as plt
import numpy as np

from src.math.vector3d import SVector3D


class Telescope:
    def __init__(self, sky_target: SVector3D, pos: SVector3D, cam_alpha, cam_beta, screen_distance=None):
        self.r = pos.r
        self.sky_r = sky_target.r
        self.target = sky_target.to_cartesian()
        self.pos = pos.to_cartesian()
        self.alpha = cam_alpha
        self.beta = cam_beta
        self.to_target = (self.target - self.pos).normalize().to_spherical()
        self.screen_distance = 2.5 * self.sky_r if screen_distance is None else screen_distance
        self.center = None
        self.corners = None
        self.up = None
        self.right = None
        self.screen_h = 0
        self.screen_w = 0
        self.update_vectors()

    def update_vectors(self):
        self.center = self.to_screen3d(self.to_target)

        screen_half_h = self.screen_distance * np.sin(self.alpha) / np.cos(self.alpha)
        screen_half_w = self.screen_distance * np.sin(self.beta) / np.cos(self.beta)
        self.up = self.to_target.to_cartesian(phi_inc=90) * screen_half_h
        self.right = self.up.cross(self.to_target).normalize() * screen_half_w
        self.screen_h = screen_half_h * 2
        self.screen_w = screen_half_w * 2
        self.corners = [
            self.center + self.up + self.right,
            self.center + self.up - self.right,
            self.center - self.up - self.right,
            self.center - self.up + self.right,
        ]

    def make_grid(self):
        grid_x = []
        grid_y = []
        grid_z = []
        for theta in np.linspace(0, 24, 12):
            for phi in np.linspace(-90, 90, 12):
                v = SVector3D(phi, theta, self.sky_r).to_cartesian()
                grid_x.append(v.x)
                grid_y.append(v.y)
                grid_z.append(v.z)
        return grid_x, grid_y, grid_z

    def to_screen3d(self, v):
        if isinstance(v, SVector3D):
            v = v.to_cartesian()
        v = v.normalize()
        point_to_screen_distance = self.screen_distance / v.calc_cos_angle(self.to_target)
        return v * point_to_screen_distance + self.pos

    def to_screen2d(self, v: SVector3D):
        dev = v - self.target.to_spherical()

        max_x = np.tan(np.pi / +4)
        min_x = np.tan(np.pi / -4)
        x = np.tan(dev.theta)
        y = np.tan(dev.phi)

        x = (x - min_x) / (max_x - min_x)
        y = (y - min_x) / (max_x - min_x)
        return x, y
        # if isinstance(v, SVector3D):
        #     v.to_cartesian()
        # x = v.distance_to_line(self.corners[1], self.corners[2]) / self.screen_w
        # y = v.distance_to_line(self.corners[0], self.corners[1]) / self.screen_h
        # return x, y

    def __inner_circle(self):
        return plt.Circle((0, 0), self.r, color='r', fill=False)

    def __outer_circle(self):
        return plt.Circle((0, 0), self.sky_r, color='b', fill=False)

    def draw(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_aspect('equal', 'datalim')
        ax2.set_aspect('equal', 'datalim')
        ax1.add_artist(self.__inner_circle())
        ax2.add_artist(self.__inner_circle())
        ax1.add_artist(self.__outer_circle())
        ax2.add_artist(self.__outer_circle())
        ax1.set_ylabel('y')
        ax2.set_ylabel('z')
        ax1.set_xlabel('x')
        ax2.set_xlabel('x')

        for corner in self.corners:
            ax1.plot([self.pos.x, corner.x], [self.pos.y, corner.y], color='green', linestyle='--', zorder=-1)
            ax2.plot([self.pos.x, corner.x], [self.pos.z, corner.z], color='green', linestyle='--', zorder=-1)

        ax1.fill([corner.x for corner in self.corners], [corner.y for corner in self.corners], edgecolor='red', facecolor=(0.5, 0, 0, 0.1))
        ax2.fill([corner.x for corner in self.corners], [corner.z for corner in self.corners], edgecolor='red', facecolor=(0.5, 0, 0, 0.1))
        ax1.scatter(self.pos.x, self.pos.y, color='red')
        ax2.scatter(self.pos.x, self.pos.z, color='red')
        ax1.scatter(self.target.x, self.target.y, color='blue')
        ax2.scatter(self.target.x, self.target.z, color='blue')
        ax1.scatter(self.center.x, self.center.y, color='cyan')
        ax2.scatter(self.center.x, self.center.z, color='cyan')

        gx, gy, gz = self.make_grid()
        ax1.scatter(gx, gy, color=(0, 0, 0, 0.1), s=2)
        ax2.scatter(gx, gz, color=(0, 0, 0, 0.1), s=2)

        ax2.set_xlim([-self.sky_r * 2, self.sky_r * 2])
        ax2.set_ylim([-self.sky_r * 2, self.sky_r * 2])
        ax1.set_xlim([-self.sky_r * 2, self.sky_r * 2])
        ax1.set_ylim([-self.sky_r * 2, self.sky_r * 2])
        plt.show()