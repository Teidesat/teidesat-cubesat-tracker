from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.catalog.starcatalog import StarCatalog
from src.image_processor.image_processor import find_stars
from src.image_processor.star_descriptor import StarDescriptor
from src.utils import time_it, Distance

# Variables
THRESHOLD = 20
PX_SENSITIVITY = 8
PATH_FRAME = Path('data/frames/video1/frame2000.jpg')
PATH_VIDEO = Path('data/videos/video1.mp4')

# Decorator
find_stars = time_it(find_stars)


def process(image, threshold, px_sensitivity):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    stars = find_stars(gray, threshold, px_sensitivity, fast=False)
    for x, y in stars:
        cv2.circle(image, (int(x), int(y)), 10, color=(0, 0, 100), thickness=1)
    return image


def single_frame_test():
    image = cv2.imread(str(PATH_FRAME)).astype('uint8')
    image = process(image, THRESHOLD, PX_SENSITIVITY)
    cv2.imshow(str(PATH_VIDEO), image)
    cv2.waitKey(0)


def video_test():
    vidcap = cv2.VideoCapture(str(PATH_VIDEO))
    success, image = vidcap.read()
    count = 0
    while success:
        image = process(image, THRESHOLD, PX_SENSITIVITY)
        cv2.imshow(str(PATH_VIDEO), image)
        cv2.waitKey(1)
        success, image = vidcap.read()
        count += 1


translator = {}
pairs = []
def find_add_candidates(descriptors, descriptor, dic):
    for desc in descriptors:
        if abs(desc.rel_dist - descriptor.rel_dist) < 0.3:
            if abs(desc.angle - descriptor.angle) < 0.3:
                dic[desc.star] += 1
                pairs.append((descriptor.star, desc.star, descriptor.first_ref, desc.first_ref, descriptor.second_ref, desc.second_ref))


def identify_test():
    # Load catalog
    FILE = Path('./data/catalog/hygdata_v3.csv')
    cat = StarCatalog(file=FILE, load_catalog=True)

    # Process image
    image = cv2.imread(str(PATH_FRAME)).astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    stars = find_stars(gray, THRESHOLD, PX_SENSITIVITY, fast=False)

    # Chose a subset
    stars = [(x, y) for x, y in stars if y < 200 and 800 < x < 1050]  # Dubhe Megrez Alioth
    # Alioth, Megrez, Dubhe
    # Phad, Merak

    stars_real = cat.get_by_names(['Alioth, Megrez, Dubhe', 'Phad', 'Merak'])

    # Match stars in the image with the stars in the database
    descs_found = StarDescriptor.build_descriptors(stars, px_radius=200, dist_func=Distance.euclidean)
    descs_original = StarDescriptor.build_descriptors(stars_real, px_radius=150, dist_func=Distance.between_spherical)

    candidates = defaultdict(lambda: defaultdict(lambda: 0))
    for desc in descs_found:
        find_add_candidates(descs_original, desc, candidates[desc.star])

    nn = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ss = '123456789'

    def translate(star):
        nonlocal nn, ss
        if star not in translator:
            if star[0] < 10:
                translator[star] = nn[-1]
                nn = nn[:-1]
            else:
                translator[star] = nn[0]
                nn = nn[1:]
        return translator[star]

    def counter(a, b):
        count = 0
        for p in pairs:
            if p[0] == a and p[1] == b:
                count += 1
        return count

    scores = []
    for p in pairs:
        scores.append(counter(p[0], p[1]) * counter(p[2], p[3]) * counter(p[4], p[5]))

    i = np.argsort(scores)
    for index in i:
        p = pairs[index]
        print('{} -> {}  |  {} -> {}  |  {} -> {}, [{}]'.format(translate(p[0]), translate(p[1]),
                                                          translate(p[2]), translate(p[3]),
                                                          translate(p[4]), translate(p[5]), scores[index]))
    for k, v in translator.items():
        print('[', k, v, ']')

    print('Candidates')
    for k, v in candidates.items():
        print(k)
        for k, v in v.items():
            s = [o[0] for o in values]
            i = s.index(k[0])
            print(' ', k, v, names[i])
    s = np.array([desc.rel_dist for desc in descs_found])
    i = np.argsort(s)
    for desc in np.array(descs_found)[i]:
        print(desc)
    for i, v in enumerate(values):
        print('{:6s} ({:7.4f}, {:7.4f})'.format(names[i], v[0], v[1]))
    s = np.array([desc.rel_dist for desc in descs_original])
    i = np.argsort(s)
    for desc in np.array(descs_original)[i]:
        print(desc)

    # values = [(cat.stars['theta'][i][0], cat.stars['phi'][i][0]) for i in indices]
    # build_descriptors(values, lambda a, b: linear_distances(a[0], b[0], a[1], b[1]), r=256)

    # dist = {}
    # for i in range(len(indices)):
    #     for j in range(i + 1, len(indices)):
    #         n = indices[i]
    #         m = indices[j]
    #         dist[(j, i)] = linear_distances(cat.stars['theta'][m], cat.stars['theta'][n], cat.stars['phi'][m], cat.stars['phi'][n])
    #
    # print(dist)
    #
    # keys = list(dist.keys())
    # for i in range(len(dist)):
    #     for j in range(len(dist)):
    #         if i != j:
    #             n = keys[i]
    #             m = keys[j]
    #             print('{} / {} = {}'.format(n, m, dist[n] / dist[m]))
    #
    # r_dist = {}
    # for i in range(len(stars)):
    #     for j in range(i + 1, len(stars)):
    #         dx = abs(stars[i][0] - stars[j][0])
    #         dy = abs(stars[i][1] - stars[j][1])
    #         r_dist[(j, i)] = np.sqrt(dx*dx + dy*dy)
    #
    # print(r_dist)
    #
    # keys = list(r_dist.keys())
    # for i in range(len(r_dist)):
    #     for j in range(len(r_dist)):
    #         if i != j:
    #             n = keys[i]
    #             m = keys[j]
    #             print('{} / {} = {}'.format(n, m, r_dist[n] / r_dist[m]))
    #
    #
    # for x, y in stars:
    #     cv2.circle(image, (int(x), int(y)), 10, color=(0, 0, 100), thickness=1)
    # cv2.imshow(str(PATH_VIDEO), image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    single_frame_test()
    # video_test()
    # identify_test()
