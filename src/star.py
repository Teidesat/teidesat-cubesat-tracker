import numpy as np
import pandas as pd


class Star:
    def __init__(self, cols, index):
        self.cols = cols
        self.index = index

    def __str__(self):
        name = self.cols['proper'][self.index]
        star_id = self.cols['id'][self.index]
        if not isinstance(name, str):
            name = 'Unnamed-{}'.format(star_id)
        return 'Star({:14s})'.format(name)

    def __getitem__(self, item):
        return self.cols[item][self.index]


class ImageStar:
    def __init__(self, x, y, ref=None):
        self.x = x
        self.y = y
        self.ref = ref  # Real Star in the database where it belongs to


class StarDB:
    def __init__(self, cols):
        self.cols = cols  # In a C++ implementation, cols will be contiguous in memory

    def __getitem__(self, item):
        if isinstance(item, int):  # Return a single Star
            return Star(self.cols, item)
        if isinstance(item, str):  # Return a single column
            return self.cols[item]
        if isinstance(item, list) and len(item) > 0:  # Return a subset of rows by index
            if isinstance(item[0], int):
                views = {col: data[item] for col, data in self.cols.items()}
                return StarDB(views)

    def __str__(self):
        key = next(self.cols.keys())
        return '{} Stars'.format(len(self.cols[key]))

    def get_first_where(self, key, equals):
        for i, row in enumerate(self.cols[key]):
            if row == equals:
                return self[i]
        return None

    def get_by_name(self, name):
        return self.get_first_where('proper', name)

    @staticmethod
    def from_csv(csv):
        df = pd.read_csv(csv)
        df['phi'] = df['ra'] * 2 * np.pi / 24
        df['theta'] = (df['dec'] + 90) * np.pi / 180
        cols = {col: df[col].to_numpy() for col in df}
        return StarDB(cols)
