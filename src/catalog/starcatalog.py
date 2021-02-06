import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.catalog.colorDic import ColorDict
from src.catalog.projections import SphereProjection, SphereEquirectProjection


class StarCatalog:
    URL = 'https://raw.githubusercontent.com/astronexus/HYG-Database/master/hygdata_v3.csv'

    def __init__(self, file, delimiter=',', telescope=None, load_catalog=False):
        self.file = Path(file)
        self.delimiter = delimiter
        self.header = None
        self.stars = {}
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

        df = pd.read_csv(self.file)
        self.header = list(df.columns)
        self.stars = {col: df[col].to_numpy() for col in df}

        # Add phi and theta from right ascension and declination for spherical coord calculation
        self.header += ['phi', 'theta']
        self.stars['phi'] = self.stars['ra'] * 2 * np.pi / 24
        self.stars['theta'] = (self.stars['dec'] + 90) * np.pi / 180
        return self

    def find_neighbours(self, star, indices_sorted_by_phi, radius):
        phi = self.stars['phi'][star]
        the = self.stars['theta'][star]
        # Neighbours according to phi values (first)
        neighbours = np.searchsorted(self.stars['phi'][indices_sorted_by_phi], (phi - radius, phi + radius))
        neighbours = indices_sorted_by_phi[neighbours[0]:neighbours[1]]
        # Neighbours according to theta values (second)
        neighbours_idx, = np.where(np.logical_and(self.stars['theta'][neighbours] > (the - radius),
                                                  self.stars['theta'][neighbours] < (the + radius)))
        return neighbours[neighbours_idx]

    def linear_distances(self, theta_from, theta_to, phi_from, phi_to):
        aux = np.sin(theta_from) * np.sin(theta_to) * np.cos(phi_from - phi_to) + np.cos(theta_from) * np.cos(theta_to)
        return 2 - 2 * aux

    def make_star_relation_db(self, min_mag=5, radius=0.12):
        """ For each selected star we calculate the relative distance to it's close neighbours """
        indices, = np.where(self.stars['mag'] < min_mag)
        indices = indices[np.argsort(self.stars['phi'][indices])]  # Sort by phi values

        index_pairs = set()
        for star in indices:
            # 1.1.) Para cada estrella, encontrar sus N vecinos en un radio
            neighbours = self.find_neighbours(star, indices, radius)
            # 1.2.) Calcular los pares de vecinos
            for i, n in enumerate(neighbours):
                for m in neighbours[i + 1:]:
                    index_pairs.add((n, m))
        print(len(index_pairs))

        # 2.) Calcular las distancias entre los vecinos
        index_pairs = np.array(list(index_pairs), copy=False, order='F')
        distances = self.linear_distances(theta_from=self.stars['theta'][index_pairs[:, 0]],
                                          theta_to=self.stars['theta'][index_pairs[:, 1]],
                                          phi_from=self.stars['phi'][index_pairs[:, 0]],
                                          phi_to=self.stars['phi'][index_pairs[:, 1]])
        print(distances)
        distances = list(distances)

        # 3.) Calcular las relaciones entre las distancias
        # ratios = []
        # points = []
        # for i, n in enumerate(zip(tqdm(index_pairs), distances)):
        #     for m in zip(index_pairs[i+1:], distances[i+1:]):
        #         pairs = (n[0], m[0])
        #         dists = (n[1], m[1])
        #         ratio = max(dists) / min(dists)
        #         ratios.append(ratio)
        #         points.append(pairs)
        # ratios = np.array(ratios)
        # points = np.array(points)
        # index = np.argsort(ratios)
        # ratios = ratios[index]
        # points = points[index]
        # np.save('./ratios.npy', ratios)
        # np.save('./points.npy', points)
        ratios = np.load('./ratios.npy')
        points = np.load('./points.npy')
        print(ratios.shape)
        print(points.shape)

        # 4.) Dadas unas relaciones intentar matchear con la lista de relaciones conocidas
        r = [3.247510358363014, 7.170167914516244, 11.4495020716726, 3.4035400159918576, 3.236534128080251, 7.477575052700445, 12.131205931525768, 1.925514283637422, 1.1339432904714808, 2.8432070945700363, 1.4931674861341762, 1.784861694950439, 2.4567638293232528, 5.218586901213926, 2.207896857373086, 3.525624496373892, 1.0480459307010475, 1.0033913531723744, 2.3025561822902687, 3.7355403348554046, 1.68656778397315, 3.6824925816023737, 1.1421997238840313, 4.849076877991599, 5.796356842596917, 1.3218650973291082, 1.6069500402900887, 1.5968248175182482, 2.106679481018741, 2.2153846153846137, 1.0428730738037313, 1.6918998377939989, 3.7237677099810083, 8.130563798219585, 2.5218591808559596, 10.706261600078149, 12.797758056982719, 2.9185417942641068, 1.373967330667302, 3.3639980778471887, 3.5375811341942707, 1.5311784891463893, 1.0595400442382368, 5.946204693970768, 12.983086053412462, 4.0269673262770365, 17.096024225847415, 20.43577767398412, 4.660399968045163, 2.193985132068849, 1.0516002245929246, 2.1969993058892636, 3.5642906722195535, 1.7676005028445434, 3.859421364985163, 1.1970777726645192, 5.082055289635635, 6.074848201774872, 1.385375336191516, 1.533282075925036, 2.310364963503649, 3.748208871420549, 1.6808673690886506, 3.670046158918565, 1.138339213580816, 4.832687527813056, 5.776765789610259, 1.31739733768866, 1.6123997754070736, 1.6223449241268018, 3.8834170778389754, 8.479146060006599, 2.629979035639414, 11.165271943820352, 13.346437282682036, 3.0436686520088663, 1.4328735334389167, 6.300241984499398, 13.756099571381474, 4.266733139029505, 18.113922264552336, 21.652524780735902, 4.9378803883104485, 2.3246151038902614, 2.183424002637654, 1.4765962105453887, 2.8751153224143384, 3.4367766879443695, 1.2759000804098248, 2.710230168407619, 3.2240356083086055, 1.316792028914721, 1.5740308267164875, 2.7858308605341247, 5.917581602373887, 4.245384389957995, 5.074731433909388, 1.1572976859372088, 1.8354578923147722, 1.1953526389537599, 3.6683598710559733, 7.792224284458338, 4.384983652498832, 9.314455861746847, 2.1241711714110725]

        all = []
        for ratio in r:
            i = np.searchsorted(ratios, ratio)
            close_by = (points[i], )
            for pair1, pair2 in close_by:
                all.extend(list(pair1) + list(pair2))

            projection = SphereEquirectProjection(width=1000)
            projection.project_grid((20, 20, 20))
            self.project(projection, min_mag=999, min_bright=1, vega_ref_bright=4., vega_ref_size=1., indices=all)
            cv2.imshow('A', projection.img)
            cv2.waitKey(1)
        cv2.waitKey(1000)

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
