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
__version__ = "0.0.3"

from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter

import cv2 as cv
import numpy as np

from src.catalog.star_catalog import StarCatalog
from src.image_processor.image_processor import (find_stars, star_tracker,
                                                 detect_blinking_star)
from src.image_processor.star_descriptor import StarDescriptor
from src.utils import time_it, Distance

#* Constants
THRESHOLD = 50
PX_SENSITIVITY = 8
FAST = True
DISTANCE = 20
BEST_ALGORITHM_INDEX = 8

SAT_DESIRED_BLINKING_FREQ = 10

PATH_FRAME = Path("./data/frames/video1/frame2000.jpg")
PATH_VIDEO = Path("./data/videos/video5.mp4")
PATH_CATALOG = Path("./data/catalog/hygdata_v3.csv")

CHECKING_VIDEO_VELOCITY = False
CHECKING_FRAME_VELOCITY = False

COLOR_CAMERA = True

#* Variables
translator = {}
pairs = []

#* Decorator
if CHECKING_FRAME_VELOCITY:
    find_stars = time_it(find_stars)
    star_tracker = time_it(star_tracker)
    detect_blinking_star = time_it(detect_blinking_star)


def main():
    """ Main function to start the program execution. """

    # single_frame_test()

    # video_test(algorithm_index=BEST_ALGORITHM_INDEX)

    #* comparison of all different find_stars' algorithms
    # for index in range(1, 9):
    #     print(f"Video test {index}...")
    #     video_test(algorithm_index=index)

    blinking_star_test(SAT_DESIRED_BLINKING_FREQ)

    # identify_test()


def process_image(image,
                  threshold: float = THRESHOLD,
                  px_sensitivity: int = PX_SENSITIVITY,
                  fast: bool = FAST,
                  distance: float = DISTANCE,
                  algorithm_index: int = BEST_ALGORITHM_INDEX):
    """ Function to process the given image and mark the detected stars. """

    if COLOR_CAMERA:
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
        start_time = perf_counter()

    wait_time = 1
    while success:
        image = process_image(image, algorithm_index=algorithm_index)

        if CHECKING_VIDEO_VELOCITY:
            processed_frames += 1
        else:
            cv.imshow(str_path_video, image)

            key = cv.waitKey(wait_time)
            if key == ord('z'):
                wait_time = 1
            if key == ord('x'):
                wait_time = 100
            if key == ord('c'):
                wait_time = 1000
            if key == ord('v'):
                wait_time = 0
            if key == ord('q'):
                break

        success, image = vidcap.read()

    if CHECKING_VIDEO_VELOCITY:
        processing_time = perf_counter() - start_time

        if CHECKING_FRAME_VELOCITY:
            print("\n  *Video process time could not be real",
                  "if also checking frame process time.*")

        print("  Processed frames:", processed_frames)
        print("  Time needed:", processing_time)
        print("  FPS:", processed_frames / processing_time)
    print()

    cv.destroyAllWindows()


def blinking_star_test(desired_blinking_freq=10):
    """ Function to test the detection of the blinking star. """

    fps = 30
    #? how do I know this value in real time?
    # if source_from_video:
    #     fps = vidcap.get(cv.CAP_PROP_FPS)
    # elif source_from_camera:
    #     fps = vidcap.get(cv.CAP_PROP_FPS) # maybe works but maybe not

    mini_test = False
    if mini_test:
        video_frame_paths = [
            str(Path("./data/images/stellarium-003.png")),
            str(Path("./data/images/stellarium-004.png")),
            str(Path("./data/images/stellarium-005.png")),
            str(Path("./data/images/stellarium-006.png")),
        ]

        video_frames = []
        for frame_path in video_frame_paths:
            frame = cv.imread(frame_path)
            if frame is None:
                sys.exit("Could not read the image.")
            else:
                video_frames.append(frame)

    else:
        str_path_video = str(PATH_VIDEO)
        vidcap = cv.VideoCapture(str_path_video)

    if CHECKING_VIDEO_VELOCITY:
        start_time = perf_counter()

    processed_frames = 0
    detected_stars = {}
    next_star_id = 0
    wait_time = 1

    while True:
        if mini_test:
            frame = video_frames[processed_frames % len(video_frames)]
        else:
            success, frame = vidcap.read()
            if not success:
                break

        if COLOR_CAMERA:
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        star_positions = find_stars(gray, THRESHOLD, PX_SENSITIVITY, FAST,
                                    DISTANCE, BEST_ALGORITHM_INDEX)
        processed_frames += 1

        detected_stars, next_star_id = star_tracker(star_positions,
                                                    detected_stars,
                                                    desired_blinking_freq, fps,
                                                    next_star_id)

        blinking_star = detect_blinking_star(detected_stars)

        if not CHECKING_VIDEO_VELOCITY:
            show_frame = frame.copy()
            for star in star_positions:
                cv.circle(
                    show_frame,
                    center=(int(star[0]), int(star[1])),
                    radius=PX_SENSITIVITY,
                    color=(0, 0, 100),
                    thickness=1,
                )
            if blinking_star is not None:
                cv.circle(
                    show_frame,
                    center=(int(blinking_star[1]["position"][0]),
                            int(blinking_star[1]["position"][1])),
                    radius=PX_SENSITIVITY,
                    color=(0, 200, 0),
                    thickness=1,
                )
            cv.imshow("blinking star", show_frame)

            key = cv.waitKey(wait_time)
            if key == ord('z'):
                wait_time = 1
            if key == ord('x'):
                wait_time = 100
            if key == ord('c'):
                wait_time = 1000
            if key == ord('v'):
                wait_time = 0
            if key == ord('q'):
                break

    if CHECKING_VIDEO_VELOCITY:
        processing_time = perf_counter() - start_time

        if CHECKING_FRAME_VELOCITY:
            print("\n  *Video process time could not be real",
                  "if also checking frame process time.*")

        print("  Processed frames:", processed_frames)
        print("  Time needed:", processing_time)
        print("  FPS:", processed_frames / processing_time)
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
    if COLOR_CAMERA:
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
