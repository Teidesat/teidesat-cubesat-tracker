import numpy as np
import colorsys


class ColorDict:
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
        color = self[key]
        return colorsys.hsv_to_rgb(color[0], color[1], color[2] * brightness)

    def hsv(self, key, brightness=1):
        color = self[key]
        return color[0], color[1], color[2] * brightness

    @staticmethod
    def random_color():
        return np.random.uniform(0, 360), 1, 1
