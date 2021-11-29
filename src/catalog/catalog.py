from pathlib import Path

import cv2

from src.catalog.projections import SphereEquirectProjection
from src.catalog.starcatalog import StarCatalog
from src.catalog.telescope import Telescope
from src.math.vector3d import SVector3D

FILE = Path('./data/catalog/hygdata_v3.csv')


if __name__ == '__main__':
    projection = SphereEquirectProjection(width=1000)
    telescope = Telescope(sky_target=SVector3D(0, 0, 1), cam_alpha=1.5, cam_beta=1.5, width=1000, height=1000)
    starMap = StarCatalog(FILE, telescope=telescope)

    starMap.load_catalog()
    projection.project_grid((20, 20, 20))
    starMap.project(projection, min_mag=4, min_bright=1, vega_ref_bright=4., vega_ref_size=1., colorize=True)
    cv2.imshow('A', projection.img)
    cv2.waitKey(0)
