#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

from pathlib import Path

import cv2 as cv

from catalog.projections import SphereEquirectProjection
from catalog.star_catalog import StarCatalog
from catalog.telescope import Telescope
from extra_math.vector3d import SVector3D

FILE = Path("./data/catalog/hygdata_v3.csv")

if __name__ == "__main__":
    projection = SphereEquirectProjection(width=1000)
    telescope = Telescope(sky_target=SVector3D(0, 0, 1),
                          cam_alpha=1.5,
                          cam_beta=1.5,
                          width=1000,
                          height=1000)
    starMap = StarCatalog(FILE, telescope=telescope)

    starMap.load_catalog()
    projection.project_grid((20, 20, 20))
    starMap.project(projection,
                    min_mag=4,
                    min_bright=1,
                    vega_ref_bright=4.,
                    vega_ref_size=1.,
                    colorize=True)
    cv.imshow('A', projection.img)
    cv.waitKey(0)
