#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TeideSat Satelite Tracking for the Optical Ground Station

# ToDo: Check if this is correct
#     This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#     This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
""" # ToDo: complete module docstring

# ToDo: add the mising module information
__authors__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
# __contact__ = "mail@example.com"
# __copyright__ = "Copyright $YEAR, $COMPANY_NAME"
__credits__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
__date__ = "2022/03/31"
__deprecated__ = False
# __email__ = "mail@example.com"
# __license__ = "GPLv3"
__maintainer__ = "Sergio Tabares Hernández"
__status__ = "Production"
__version__ = "0.0.2"

import sys

from collections import defaultdict
from pathlib import Path
from time import time

import cv2 as cv
import numpy as np

from src.catalog.star_catalog import StarCatalog
from src.image_processor.image_processor import find_stars
from src.image_processor.star_descriptor import StarDescriptor
from src.utils import time_it, Distance

#* Constants
THRESHOLD = 20
PX_SENSITIVITY = 8
FAST = True
DISTANCE = 20
BEST_ALGORITHM_INDEX = 8

PATH_FRAME = Path("./data/frames/video1/frame2000.jpg")
PATH_VIDEO = Path("./data/videos/video1.mp4")
PATH_CATALOG = Path("./data/catalog/hygdata_v3.csv")

CHECKING_VIDEO_VELOCITY = False
CHECKING_FRAME_VELOCITY = True

#* Variables
translator = {}
pairs = []

#* Decorator
if CHECKING_FRAME_VELOCITY:
    find_stars = time_it(find_stars)


def main():
    """ Main function to start the program execution. """

    # single_frame_test()

    # video_test(algorithm_index=BEST_ALGORITHM_INDEX)

    #* comparison of all different find_stars' algorithms
    for index in range(1, 9):
        print(f"Video test {index}...")
        video_test(algorithm_index=index)

    # identify_test()


def process_image(image,
                  threshold: float = THRESHOLD,
                  px_sensitivity: int = PX_SENSITIVITY,
                  fast: bool = FAST,
                  distance: float = DISTANCE,
                  algorithm_index: int = BEST_ALGORITHM_INDEX):
    """ Function to process the given image and mark the detected stars. """

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    stars = find_stars(gray, threshold, px_sensitivity, fast, distance,
                       algorithm_index)

    if not CHECKING_VIDEO_VELOCITY:
        for x_coord, y_coord in stars:
            cv.circle(
                image,
                center=(int(x_coord), int(y_coord)),
                radius=PX_SENSITIVITY,
                color=(0, 0, 100),
                thickness=1,
            )
    return image


def single_frame_test(
        str_path_frame=str(PATH_FRAME), algorithm_index=BEST_ALGORITHM_INDEX):
    """ Function to test the implemented processing image methods with a single
    video frame or image. """

    print("Processing frame from:", str_path_frame)
    image = cv.imread(str_path_frame)
    if image is None:
        sys.exit("Could not read the image.")

    image = process_image(image, algorithm_index=algorithm_index)
    print("\n")

    cv.imshow(str_path_frame, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video_test(
        str_path_video=str(PATH_VIDEO), algorithm_index=BEST_ALGORITHM_INDEX):
    """ Function to test the implemented processing image methods with a whole
    video. """

    print("Processing video from:", str_path_video)
    vidcap = cv.VideoCapture(str_path_video)
    success, image = vidcap.read()

    if CHECKING_VIDEO_VELOCITY:
        processed_frames = 0
        start_time = time()

    while success:
        image = process_image(image, algorithm_index=algorithm_index)

        if CHECKING_VIDEO_VELOCITY:
            processed_frames += 1
        else:
            cv.imshow(str_path_video, image)

            key = cv.waitKey(1)
            if key == ord('q') or key == ord('s'):
                break

        success, image = vidcap.read()

    print("                                                        ", end="\r")
    if CHECKING_VIDEO_VELOCITY:
        end_time = time()
        process_time = end_time - start_time

        if CHECKING_FRAME_VELOCITY:
            print("  *Video process time could not be real",
                  "if also checking frame process time.*")

        print("  Processed frames:", processed_frames)
        print("  Time needed:", process_time)
        print("  FPS:", processed_frames / process_time)
    print()

    cv.destroyAllWindows()


def find_add_candidates(descriptors, descriptor, dic):
    """ Docstring """  # ToDo: redact docstring

    for desc in descriptors:
        if abs(desc.rel_dist - descriptor.rel_dist) < 0.3:
            if abs(desc.angle - descriptor.angle) < 0.3:
                dic[desc.star] += 1
                pairs.append((
                    descriptor.star,
                    desc.star,
                    descriptor.first_ref,
                    desc.first_ref,
                    descriptor.second_ref,
                    desc.second_ref,
                ))


def identify_test():
    """ Docstring """  # ToDo: redact docstring

    # Load catalog
    cat = StarCatalog(file=PATH_CATALOG, load_catalog=True)

    # Process image
    image = cv.imread(str(PATH_FRAME)).astype("uint8")
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    stars = find_stars(gray, THRESHOLD, PX_SENSITIVITY, fast=False)

    # Chose a subset
    stars = [(x, y) for x, y in stars
             if y < 200 and 800 < x < 1050]  # Dubhe Megrez Alioth
    # Alioth, Megrez, Dubhe
    # Phad, Merak

    stars_real = cat.get_by_names(["Alioth, Megrez, Dubhe", "Phad", "Merak"])

    # Match stars in the image with the stars in the database
    descs_found = StarDescriptor.build_descriptors(
        stars, px_radius=200, dist_func=Distance.euclidean)
    descs_original = StarDescriptor.build_descriptors(
        stars_real, px_radius=150, dist_func=Distance.between_spherical)

    candidates = defaultdict(lambda: defaultdict(lambda: 0))
    for desc in descs_found:
        find_add_candidates(descs_original, desc, candidates[desc.star])

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers = "123456789"  #! Numbers not used

    def translate(star):
        """ Docstring """  # ToDo: redact docstring

        nonlocal letters, numbers  #! Numbers not used
        if star not in translator:
            if star[0] < 10:
                translator[star] = letters[-1]
                letters = letters[:-1]
            else:
                translator[star] = letters[0]
                letters = letters[1:]
        return translator[star]

    def counter(first_comp, second_comp):
        """ Docstring """  # ToDo: redact docstring

        count = 0
        for pair in pairs:
            if pair[0] == first_comp and pair[1] == second_comp:
                count += 1
        return count

    scores = []
    for pair in pairs:
        scores.append(
            counter(pair[0], pair[1]) * counter(pair[2], pair[3]) *
            counter(pair[4], pair[5]))

    i = np.argsort(scores)
    for index in i:
        pair = pairs[index]
        print(
            f"{translate(pair[0])} -> {translate(pair[1])}  |  ",
            f"{translate(pair[2])} -> {translate(pair[3])}  |  ",
            f"{translate(pair[4])} -> {translate(pair[5])}, [{scores[index]}]")

    for key, value in translator.items():
        print(f"[{key}, {value}]")

    # print("Candidates")
    # for k, v in candidates.items():
    #     print(k)
    #     for k, v in v.items():
    #         s = [o[0] for o in values]
    #         i = s.index(k[0])
    #         print(' ', k, v, names[i])
    # s = np.array([desc.rel_dist for desc in descs_found])
    # i = np.argsort(s)
    # for desc in np.array(descs_found)[i]:
    #     print(desc)
    # for i, v in enumerate(values):
    #     print(f"{names[i]} ({v[0]}, {v[1]})")
    # s = np.array([desc.rel_dist for desc in descs_original])
    # i = np.argsort(s)
    # for desc in np.array(descs_original)[i]:
    #     print(desc)

    # values = [(cat.stars["theta"][i][0], cat.stars["phi"][i][0]) for i in indices]
    # build_descriptors(values, lambda a, b: linear_distances(a[0], b[0], a[1], b[1]), r=256)

    # dist = {}
    # for i in range(len(indices)):
    #     for j in range(i + 1, len(indices)):
    #         n = indices[i]
    #         m = indices[j]
    #         dist[(j, i)] = linear_distances(cat.stars["theta"][m],
    #                                         cat.stars["theta"][n],
    #                                         cat.stars["phi"][m],
    #                                         cat.stars["phi"][n])

    # print(dist)
    #
    # keys = list(dist.keys())
    # for i in range(len(dist)):
    #     for j in range(len(dist)):
    #         if i != j:
    #             n = keys[i]
    #             m = keys[j]
    #             print(f"{n} / {m} = {dist[n] / dist[m]}")
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
    #             print(f"{n} / {m} = {r_dist[n] / r_dist[m]}")
    #
    #
    # for x, y in stars:
    #     cv.circle(image, (int(x), int(y)), 10, color=(0, 0, 100), thickness=1)
    # cv.imshow(str(PATH_VIDEO), image)
    # cv.waitKey(0)


if __name__ == "__main__":
    main()
