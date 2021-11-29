import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.catalog.colorDic import ColorDict
from src.catalog.projections import SphereProjection
from src.catalog.star import Star
from src.utils import time_it


class StarCatalog:
    URL = 'https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv'

    def __init__(self, file, telescope=None, load_catalog=False):
        self.file = Path(file)
        self.telescope = telescope
        self._cols = None
        self._list = None
        if load_catalog:
            self.load_catalog()

    def __getitem__(self, item):
        if isinstance(item, int):  # Return a single Star
            return Star(self._cols, item)
        if isinstance(item, str):  # Return a single column
            return self._cols[item]
        if isinstance(item, list) and len(item) > 0:  # Return a subset of rows by index
            if isinstance(item[0], int):
                view = StarCatalog(self.file, self.telescope)
                view._cols = {col: data[item] for col, data in self._cols.items()}
                return view

    def __str__(self):
        return '{} Stars'.format(len(self))

    def __len__(self):
        key = list(self._cols.keys())[0]
        return len(self._cols[key])

    def as_list(self):
        if self._list is None:
            self._list = [self[i] for i in range(len(self))]
        return self._list

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

        df = pd.read_csv(self.file)
        df['phi'] = df['ra'] * 2 * np.pi / 24
        df['the'] = (df['dec'] + 90) * np.pi / 180
        df['descriptors'] = [None] * len(df)
        self._cols = {col: df[col].to_numpy() for col in df}
        return self

    def get_by_name(self, name):
        for i, row in enumerate(self._cols['proper']):
            if row == name:
                return self[i]
        return None

    def get_by_names(self, names):
        stars = []
        for i, row in enumerate(self._cols['proper']):
            if row in names:
                stars.append(self[i])
        return stars

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

        phi_arr = self['phi'] if indices is None else self['phi'][indices]
        the_arr = self['the'] if indices is None else self['the'][indices]
        mag_arr = self['mag'] if indices is None else self['mag'][indices]
        con_arr = self['con'] if indices is None else self['con'][indices]

        x_arr, y_arr = projection.from_spherical_int(phi_arr, the_arr)

        for mag, con, x, y in zip(mag_arr, con_arr, x_arr, y_arr):
            if min_mag is None or mag < min_mag:
                bright = StarCatalog.calc_bright(mag, vega_ref_bright, val_max=1, val_min=min_bright) * 255
                radius = StarCatalog.calc_size(mag, vega_ref_size, val_max=max_radius, val_min=1)
                color = colors.rgb(con, bright) if (colorize and con) else (bright, bright, bright)
                projection.fill_circle((x, y), radius, color)
