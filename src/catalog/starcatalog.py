import urllib.request
from pathlib import Path

import numpy as np

from src.catalog.colorDic import ColorDict
from src.catalog.projections import SphereProjection
from src.star import StarDB


class StarCatalog:
    URL = 'https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv'

    def __init__(self, file, telescope=None, load_catalog=False):
        self.file = Path(file)
        self.stars = None
        self.telescope = telescope
        if load_catalog:
            self.load_catalog()

    def load_catalog(self):
        """ Open the csv catalog if catalog is not in local, download it """
        if self.file.is_dir():
            raise Exception('{} is dir !'.format(self.file))
        if StarCatalog.URL is None:
            raise Exception('{} not found, need url to download !'.format(self.file))

        if not self.file.exists() or not self.file.is_file():
            print('Downloading... {}'.format(StarCatalog.URL))
            txt = urllib.request.urlopen(StarCatalog.URL).read().decode('utf8')
            self.file.absolute().parent.mkdir(exist_ok=True, parents=True)
            self.file.write_text(txt)
            print('Done!')

        self.stars = StarDB.from_csv(self.file)
        return self

    def __getitem__(self, item):
        return self.stars[item]

    # Drawing ----------------------------------------------------------------------------------------------------------
    @staticmethod
    def calc_bright(magnitude, vega_ref=1., val_max=None, val_min=None):
        return np.clip(vega_ref * np.power(10, 0.4 * -magnitude), val_min, val_max)

    @staticmethod
    def calc_size(magnitude, vega_ref=1., val_max=None, val_min=None):
        return np.clip(vega_ref * np.power(10, 0.4 * -magnitude), val_min, val_max)

    def project(self, projection: SphereProjection,
                min_mag=None, vega_ref_bright=4., min_bright=0,
                vega_ref_size=1., max_radius=3,
                colorize=False, indices=None):
        colors = ColorDict()
        phi_arr = self.stars['phi'] if indices is None else self.stars['phi'][indices]
        the_arr = self.stars['theta'] if indices is None else self.stars['theta'][indices]
        mag_arr = self.stars['mag'] if indices is None else self.stars['mag'][indices]
        con_arr = self.stars['con'] if indices is None else self.stars['con'][indices]

        x_arr, y_arr = projection.from_spherical_int(phi_arr, the_arr)

        for mag, con, x, y in zip(mag_arr, con_arr, x_arr, y_arr):
            if min_mag is None or mag < min_mag:
                bright = StarCatalog.calc_bright(mag, vega_ref_bright, val_max=1, val_min=min_bright) * 255
                radius = StarCatalog.calc_size(mag, vega_ref_size, val_max=max_radius, val_min=1)
                color = colors.rgb(con, bright) if (colorize and con) else (bright, bright, bright)
                projection.fill_circle((x, y), radius, color)
