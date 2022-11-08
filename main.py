#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TeideSat Satellite Tracking for the Optical Ground Station

# ToDo: Check if this is correct
#     This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#     This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.
"""  # ToDo: complete module docstring

# ToDo: add the missing module information
__authors__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
# __contact__ = "mail@example.com"
# __copyright__ = "Copyright $YEAR, $COMPANY_NAME"
__credits__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
__date__ = "2022/06/12"
__deprecated__ = False
# __email__ = "mail@example.com"
# __license__ = "GPLv3"
__maintainer__ = "Sergio Tabares Hernández"
__status__ = "Production"
__version__ = "0.0.5"

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import cv2 as cv
import numpy as np

from src.catalog.star_catalog import StarCatalog
from src.image_processor.image_processor import (detect_stars, track_stars,
                                                 detect_blinking_star,
                                                 detect_shooting_stars)
from src.image_processor.star_descriptor import StarDescriptor
from src.utils import time_it, Distance

# * Constants
STAR_DETECTOR_THRESHOLD = 50
FAST = True
MIN_PRUNE_DISTANCE = 20.0

SAT_DESIRED_BLINKING_FREQ = 15.0
MOVEMENT_THRESHOLD = 3.0
PX_SENSITIVITY = 8

PATH_FRAME = Path("./data/images/original.jpg")
PATH_VIDEO = Path("./data/videos/video4.mp4")
PATH_CATALOG = Path("./data/catalog/hygdata_v3.csv")
PATH_SAT_LOG = Path("./data/logs/satellite_log.csv")

CHECKING_VIDEO_VELOCITY = False
CHECKING_FRAME_VELOCITY = False

COLOR_CAMERA = True
VIDEO_FROM_CAMERA = False

COLORIZED_TRACKED_STARS = False
OUTPUT_VIDEO_TO_FILE = True
PATH_OUTPUT_VIDEO = Path("./data/videos/video_output_" +
                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")

# * Global variables
translator = {}
pairs = []

# * Decorators
if CHECKING_FRAME_VELOCITY:
    detect_stars = time_it(detect_stars)
    star_tracker = time_it(track_stars)
    detect_shooting_stars = time_it(detect_shooting_stars)
    detect_blinking_star = time_it(detect_blinking_star)


def main():
    """ Main function to start the program execution. """

    # single_frame_test()

    # video_test()

    satellite_detection_test()

    # identify_test()


def process_image(
        image,
        color_camera: bool = COLOR_CAMERA,
        star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
        fast: bool = FAST,
        min_prune_distance: float = MIN_PRUNE_DISTANCE,
        checking_video_velocity: bool = CHECKING_VIDEO_VELOCITY,
):
    """ Function to process the given image and mark the detected stars. """

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if color_camera else image

    star_detector = cv.FastFeatureDetector_create(
        threshold=star_detector_threshold)

    stars = detect_stars(gray, star_detector, fast, min_prune_distance)

    if not checking_video_velocity:
        image = draw_found_stars(image, stars)

    return image


def single_frame_test(str_path_frame=str(PATH_FRAME)):
    """ Function to test the implemented processing image methods with a single
    video frame or image. """

    print("Processing frame from:", str_path_frame)
    image = cv.imread(str_path_frame)
    if image is None:
        sys.exit("\nCould not read the image.")

    image = process_image(image)
    print("\n")

    cv.imshow(str_path_frame, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def video_test(str_path_video=str(PATH_VIDEO)):
    """ Function to test the implemented processing image methods with a whole
    video. """

    print("Processing video from:", str_path_video)
    vid_cap = cv.VideoCapture(str_path_video)
    if not vid_cap.isOpened():
        sys.exit("\nError: Unable to open video.")

    wait_time = 1
    wait_options = {
        ord('z'): 1,
        ord('x'): 100,
        ord('c'): 1000,
        ord('v'): 0,
    }

    if CHECKING_VIDEO_VELOCITY:
        processed_frames = 0
        start_time = perf_counter()
    else:
        cv.namedWindow(str_path_video, cv.WINDOW_NORMAL)

        if OUTPUT_VIDEO_TO_FILE:
            output_video = create_export_video_file(vid_cap)

    while True:
        success, frame = vid_cap.read()
        if not success:
            break

        show_frame = process_image(frame)

        if CHECKING_VIDEO_VELOCITY:
            processed_frames += 1
        else:
            cv.imshow(str_path_video, show_frame)

            if OUTPUT_VIDEO_TO_FILE:
                output_video.write(show_frame)

            key = cv.waitKey(wait_time)

            if key == ord('q'):
                break

            wait_time = wait_options.get(key, wait_time)

    if CHECKING_VIDEO_VELOCITY:
        print_time_statistics(processed_frames, start_time)

    elif OUTPUT_VIDEO_TO_FILE:
        print(f"Video saved on '{str(PATH_OUTPUT_VIDEO)}'")

    cv.destroyAllWindows()


def satellite_detection_test(
        sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
        star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
        fast: bool = FAST,
        min_prune_distance: float = MIN_PRUNE_DISTANCE,
        movement_threshold: float = MOVEMENT_THRESHOLD,
        video_from_camera: bool = VIDEO_FROM_CAMERA,
        color_camera: bool = COLOR_CAMERA,
        checking_video_velocity: bool = CHECKING_VIDEO_VELOCITY,
        output_video_to_file: bool = OUTPUT_VIDEO_TO_FILE,
):
    """ Function to test the detection of the blinking star. """

    if video_from_camera:
        video_path = 0  # Default webcam id
        print("Processing video from camera number ", video_path)

    else:
        video_path = str(PATH_VIDEO)
        print("Processing video from:", video_path)

    vid_cap = cv.VideoCapture(video_path)
    if not vid_cap.isOpened():
        sys.exit("\nError: Unable to open video.")

    # if VIDEO_FROM_CAMERA this could not work
    video_fps = vid_cap.get(cv.CAP_PROP_FPS)

    star_detector = cv.FastFeatureDetector_create(
        threshold=star_detector_threshold)

    tracked_stars = {}
    satellite_log = []

    wait_time = 1
    wait_options = {
        ord('z'): 1,
        ord('x'): 100,
        ord('c'): 1000,
        ord('v'): 0,
    }

    if checking_video_velocity:
        processed_frames = 0
        start_time = perf_counter()
    else:
        cv.namedWindow("Satellite detection", cv.WINDOW_NORMAL)

        if output_video_to_file:
            output_video = create_export_video_file(vid_cap)

    while True:
        success, frame = vid_cap.read()
        if not success:
            break

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) if color_camera else frame

        new_star_positions = detect_stars(gray,
                                          star_detector,
                                          fast,
                                          min_prune_distance)

        track_stars(new_star_positions,
                    tracked_stars,
                    sat_desired_blinking_freq,
                    video_fps)

        shooting_stars = detect_shooting_stars(tracked_stars,
                                               movement_threshold)

        satellite = detect_blinking_star(shooting_stars)

        if checking_video_velocity:
            processed_frames += 1

        else:
            show_frame = frame.copy()

            # show_frame = draw_found_stars(show_frame, new_star_positions)
            show_frame = draw_tracked_stars(show_frame, tracked_stars)
            show_frame = draw_shooting_stars(show_frame, shooting_stars)

            if satellite is not None:
                satellite_log.append(deepcopy(satellite))
                show_frame = draw_satellite(show_frame, satellite)

            cv.imshow("Satellite detection", show_frame)

            if output_video_to_file:
                output_video.write(show_frame)

            key = cv.waitKey(wait_time)

            if key == ord('q'):
                break

            wait_time = wait_options.get(key, wait_time)

    if checking_video_velocity:
        print_time_statistics(processed_frames, start_time)
    else:
        export_satellite_log(satellite_log)

        if output_video_to_file:
            print(f"Video saved on '{str(PATH_OUTPUT_VIDEO)}'")

    cv.destroyAllWindows()


def print_time_statistics(
        processed_frames: int,
        start_time: float,
):
    """ Function to print processing time statistics. """

    processing_time = perf_counter() - start_time

    if CHECKING_FRAME_VELOCITY:
        print("\n  *Video process time could not be real",
              "if also checking frame process time.*")

    print("  Processed frames:", processed_frames)
    print("  Time needed:", processing_time)
    print("  FPS:", processed_frames / processing_time)
    print()


def draw_found_stars(
        show_frame,
        found_stars: list[tuple[int, int]],
        radius: int = PX_SENSITIVITY,
        color: tuple = (0, 0, 100),
        thickness: int = 2,
):
    """ Function to draw in the given frame a circle around every found
    star. """

    for star in found_stars:
        cv.circle(
            show_frame,
            center=(int(star[0]), int(star[1])),
            radius=radius,
            color=color,
            thickness=thickness,
        )

    return show_frame


def draw_tracked_stars(
        show_frame,
        tracked_stars: dict[int, dict],
        radius: int = PX_SENSITIVITY,
        color: tuple = None,
        thickness: int = 2,
        colorized_tracked_stars: bool = COLORIZED_TRACKED_STARS,
):
    """ Function to draw in the given frame a circle around every tracked
    star. """

    for star in tracked_stars.values():

        if color is None:
            if colorized_tracked_stars:
                draw_color = star["color"]
            else:
                draw_color = (0, 0, 100)
        else:
            draw_color = color

        cv.circle(
            show_frame,
            center=(int(star["last_positions"][-1][0]),
                    int(star["last_positions"][-1][1])),
            radius=radius,
            color=draw_color,
            thickness=thickness,
        )

    return show_frame


def draw_shooting_stars(
        show_frame,
        shooting_stars: dict[int, dict],
        radius: int = PX_SENSITIVITY,
        color: tuple = (0, 200, 200),
        thickness: int = 2,
):
    """ Function to draw in the given frame a circle around every shooting
    star. """

    for star in shooting_stars.values():
        cv.circle(
            show_frame,
            center=(int(star["last_positions"][-1][0]),
                    int(star["last_positions"][-1][1])),
            radius=radius,
            color=color,
            thickness=thickness,
        )

    return show_frame


def draw_satellite(
        show_frame,
        satellite: tuple[int, dict],
        radius: int = PX_SENSITIVITY,
        color: tuple = (0, 200, 0),
        thickness: int = 2,
):
    """ Function to draw in the given frame a circle around the satellite
    detected. """

    cv.circle(
        show_frame,
        center=(int(satellite[1]["last_positions"][-1][0]),
                int(satellite[1]["last_positions"][-1][1])),
        radius=radius,
        color=color,
        thickness=thickness,
    )

    return show_frame


def create_export_video_file(vid_cap):
    """ Function to create a video file to export the processed frames. """

    width = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid_cap.get(cv.CAP_PROP_FPS))
    frame_count = int(vid_cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo format: {width}x{height} px - {fps} fps")
    print("Number of frames:", frame_count)

    output_video = cv.VideoWriter(
        str(PATH_OUTPUT_VIDEO),
        cv.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    if not output_video.isOpened():
        sys.exit("\nError: Unable to create video file.")

    return output_video


def export_satellite_log(satellite_log: list[tuple[int, dict]]):
    """ Function to export the satellite log into a file. """

    with open(str(PATH_SAT_LOG), "w", encoding="utf-8-sig") as file:
        print("id;",
              "last_times_detected;",
              "lifetime;",
              "left_lifetime;",
              "detection_confidence;",
              "blinking_freq;",
              "movement_vector;",
              "last_positions;",
              file=file)

        for star_id, star_info in satellite_log:
            print(f"{star_id};",
                  f"{star_info['last_times_detected']};",
                  f"{star_info['lifetime']};",
                  f"{star_info['left_lifetime']};",
                  f"{star_info['detection_confidence']};",
                  f"{star_info['blinking_freq']};",
                  f"{star_info['movement_vector']};",
                  f"{star_info['last_positions']};",
                  file=file)


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


def identify_test(
        color_camera=COLOR_CAMERA,
        star_detector_threshold=STAR_DETECTOR_THRESHOLD,
):
    """ Docstring """  # ToDo: redact docstring

    # Load catalog
    cat = StarCatalog(file=PATH_CATALOG, load_catalog=True)

    # Process image
    image = cv.imread(str(PATH_FRAME)).astype("uint8")

    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if color_camera else image

    star_detector = cv.FastFeatureDetector_create(
        threshold=star_detector_threshold)

    stars = detect_stars(gray, star_detector, fast=False)

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
    numbers = "123456789"  # ! Numbers not used

    def translate(star):
        """ Docstring """  # ToDo: redact docstring

        nonlocal letters, numbers  # ! Numbers not used
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
