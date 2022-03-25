#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from catalog.color_dict import ColorDict
from catalog.projections import SphereProjection
from catalog.star import Star


class StarCatalog:
    """ Class docstring """  # ToDo: redact docstring

    URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv"

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
        if isinstance(
                item,
                list) and len(item) > 0:  # Return a subset of rows by index
            if isinstance(item[0], int):
                view = StarCatalog(self.file, self.telescope)
                view._cols = {
                    col: data[item]
                    for col, data in self._cols.items()
                }
                return view

    def __str__(self):
        return f"{len(self)} Stars"

    def __len__(self):
        key = list(self._cols.keys())[0]
        return len(self._cols[key])

    def as_list(self):
        """ Docstring """  # ToDo: redact docstring

        if self._list is None:
            self._list = [self[i] for i in range(len(self))]
        return self._list

    def load_catalog(self):
        """ Open the csv catalog. If catalog is not in local, download it."""

        if self.file.is_dir():
            raise Exception(f"{self.file} is dir!")
        if StarCatalog.URL is None:
            raise Exception(f"{self.file} not found, need url to download!")

        if not self.file.exists() or not self.file.is_file():
            print(f"Downloading... {StarCatalog.URL}")
            txt = urllib.request.urlopen(StarCatalog.URL).read().decode("utf8")
            self.file.absolute().parent.mkdir(exist_ok=True, parents=True)
            self.file.write_text(txt)
            print("Done!")

        file_data = pd.read_csv(self.file)
        file_data["phi"] = file_data["ra"] * 2 * np.pi / 24
        file_data["the"] = (file_data["dec"] + 90) * np.pi / 180
        file_data["descriptors"] = [None] * len(file_data)
        self._cols = {col: file_data[col].to_numpy() for col in file_data}
        return self

    def get_by_name(self, name):
        """ Docstring """  # ToDo: redact docstring

        for i, row in enumerate(self._cols["proper"]):
            if row == name:
                return self[i]
        return None

    def get_by_names(self, names):
        """ Docstring """  # ToDo: redact docstring

        stars = []
        for i, row in enumerate(self._cols["proper"]):
            if row in names:
                stars.append(self[i])
        return stars

    # Drawing -----------------------------------------------------------------
    @staticmethod
    def calc_bright(magnitude, vega_ref=1., val_max=None, val_min=None):
        """ Docstring """  # ToDo: redact docstring

        return np.clip(
            vega_ref * np.power(10, 0.4 * -magnitude),
            val_min,
            val_max,
        )

    @staticmethod
    def calc_size(magnitude, vega_ref=1., val_max=None, val_min=None):
        """ Docstring """  # ToDo: redact docstring

        return np.clip(
            vega_ref * np.power(10, 0.4 * -magnitude),
            val_min,
            val_max,
        )

    def project(self,
                projection: SphereProjection,
                min_mag=None,
                vega_ref_bright=4.,
                min_bright=0,
                vega_ref_size=1.,
                max_radius=3,
                colorize=False,
                indices=None):
        """ Docstring """  # ToDo: redact docstring

        colors = ColorDict()

        if indices is None:
            phi_arr = self["phi"]
            the_arr = self["the"]
            mag_arr = self["mag"]
            con_arr = self["con"]
        else:
            phi_arr = self["phi"][indices]
            phi_arr = self["the"][indices]
            phi_arr = self["mag"][indices]
            phi_arr = self["con"][indices]

        x_arr, y_arr = projection.from_spherical_int(phi_arr, the_arr)

        for mag, con, x_coord, y_coord in zip(mag_arr, con_arr, x_arr, y_arr):
            if min_mag is None or mag < min_mag:
                bright = (StarCatalog.calc_bright(
                    mag,
                    vega_ref_bright,
                    val_max=1,
                    val_min=min_bright,
                ) * 255)
                radius = StarCatalog.calc_size(mag,
                                               vega_ref_size,
                                               val_max=max_radius,
                                               val_min=1)

                if (colorize and con):
                    color = colors.rgb(con, bright)
                else:
                    color = (bright, bright, bright)

                projection.fill_circle((x_coord, y_coord), radius, color)
