#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions and star
detection algorithms.
"""

from colorsys import hsv_to_rgb
from math import inf, dist
import random
from time import time

import cv2 as cv
import numpy as np
import skimage.feature

#* Constants
THRESHOLD = 50
PX_SENSITIVITY = 8
FAST = True
DISTANCE = 20
BEST_ALGORITHM_INDEX = 8

DEFAULT_LEFT_LIFETIME = 10
FREQ_THRESHOLD = 3

KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

random.seed(time())


def prune_close_points(indices: list[tuple[int, int]],
                       min_distance: float) -> list[tuple[int, int]]:
    """ Function to prune close points because of their probabilities of being a
    duplicate reference to the same element of an image. """

    pruned = []
    for i, point_1 in enumerate(indices):
        duplicate = False
        for point_2 in indices[i + 1:]:
            if dist(point_1, point_2) < min_distance:
                duplicate = True
                break
        if not duplicate:
            pruned.append(point_1)
    return pruned


def find_stars(
        image,
        threshold: float = THRESHOLD,
        px_sensitivity: int = PX_SENSITIVITY,
        fast: bool = FAST,
        distance: float = DISTANCE,
        algorithm_index: int = BEST_ALGORITHM_INDEX) -> list[tuple[int, int]]:
    """
    Function to get all the bright points of a given image.

    To do so, you can select one of the following algorithms by specifying it's
    index on the following list:

    1. Sobel filter.
    2. Adaptive threshold.
    3. Scikit-Image's Laplacian of Gaussian (LoG).
    4. Scikit-Image's Difference of Gaussian (DoG).
    5. Scikit-Image's Determinant of Hessian (DoH).
    6. OpenCV's Simple Blob Detector.
    7. OpenCV's Scale-Invariant Feature Transform (SIFT).
    8. OpenCV's Features from Accelerated Segment Test (FAST).
    """

    algorithms_list = [
        sobel_filter,
        adaptative_threshold,
        scikit_blob_log,
        scikit_blob_dog,
        scikit_blob_doh,
        opencv_blob,
        opencv_sift,
        opencv_fast,
    ]

    return algorithms_list[algorithm_index - 1](image, threshold,
                                                px_sensitivity, fast, distance)


def sobel_filter(image, threshold: float, px_sensitivity: int, fast: bool,
                 distance: float) -> list[tuple[int, int]]:
    """ Sobel filter's implementation for star detection. """

    edges_x = cv.filter2D(image, cv.CV_8U, KERNEL_X)
    edges_y = cv.filter2D(image, cv.CV_8U, KERNEL_Y)
    mask = np.minimum(edges_x, edges_y)
    indices = np.argwhere(mask > threshold)

    if fast:
        # A) May give two (or more?) detections for one star
        # B) Not smooth transitions between frames

        # Group stars that are closer
        indices = np.round(indices / px_sensitivity) * px_sensitivity
        indices = {(x, y) for y, x in indices}
        return indices

    else:
        # Same as before but now store the original index to recover later the
        #  original coordinates
        # Then prune the positions that are too close

        indices_round = np.round(indices / px_sensitivity) * px_sensitivity
        indices_round = {(y, x): i for i, (y, x) in enumerate(indices_round)}
        indices = [(indices[i][1], indices[i][0])
                   for _, i in indices_round.items()]
        return prune_close_points(indices, distance)


def adaptative_threshold(image, threshold: float, px_sensitivity: int,
                         fast: bool, distance: float) -> list[tuple[int, int]]:
    """ Adaptive threshold's implementation for star detection. """

    if distance % 2 == 0:
        distance += 1

    # bnw = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv.THRESH_BINARY, distance, -threshold)
    bnw = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 9, -40)

    indices = np.where(bnw)
    points = zip(indices[1], indices[0])

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def scikit_blob_log(image, threshold: float, px_sensitivity: int, fast: bool,
                    distance: float) -> list[tuple[int, int]]:
    """ Scikit-Image's Laplacian of Gaussian algorithm for image detection. """

    # blobs_log = skimage.feature.blob_log(image, threshold, max_sigma=2)
    blobs_log = skimage.feature.blob_log(image, threshold=0.075, max_sigma=2)
    points = np.flip(np.delete(blobs_log, 2, axis=1))

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def scikit_blob_dog(image, threshold: float, px_sensitivity: int, fast: bool,
                    distance: float) -> list[tuple[int, int]]:
    """ Scikit-Image's Difference of Gaussian algorithm for image detection. """

    # blobs_dog = skimage.feature.blob_dog(image, threshold, max_sigma=2)
    blobs_dog = skimage.feature.blob_dog(image, threshold=0.05, max_sigma=2)
    points = np.flip(np.delete(blobs_dog, 2, axis=1))

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def scikit_blob_doh(image, threshold: float, px_sensitivity: int, fast: bool,
                    distance: float) -> list[tuple[int, int]]:
    """ Scikit-Image's Determinant of Hessian algorithm for image detection. """

    # blobs_doh = skimage.feature.blob_doh(image, threshold, max_sigma=2)
    blobs_doh = skimage.feature.blob_doh(image, threshold=0.00075, max_sigma=2)
    points = np.flip(np.delete(blobs_doh, 2, axis=1))

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def opencv_blob(image, threshold: float, px_sensitivity: int, fast: bool,
                distance: float) -> list[tuple[int, int]]:
    """ OpenCV's Simple Blob Detector algorithm. """

    # Setup Simple Blob Detector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 50

    # Filter by Area.
    params.filterByArea = False
    # params.minArea = 5
    # params.maxArea = 20

    # Filter by Circularity
    # params.filterByCircularity = False
    # params.minCircularity = 0.7
    # params.maxCircularity = 1

    # Filter by Convexity
    # params.filterByConvexity = False
    # params.minConvexity = 0.87

    # Filter by Inertia
    # params.filterByInertia = False
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    # OLD: detector = cv.SimpleBlobDetector(params)
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image)
    points = [keypoint.pt for keypoint in keypoints]

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def opencv_sift(image, threshold: float, px_sensitivity: int, fast: bool,
                distance: float) -> list[tuple[int, int]]:
    """ OpenCV's Scale-Invariant Feature Transform (SIFT) algorithm for star
    detection. """

    sift = cv.SIFT_create(nOctaveLayers=1,
                          contrastThreshold=0.1,
                          edgeThreshold=3,
                          sigma=0.2)
    keypoints = sift.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    if not fast:
        points = prune_close_points(list(points), distance)

    return points


def opencv_fast(image, threshold: float, px_sensitivity: int, fast: bool,
                distance: float) -> list[tuple[int, int]]:
    """ OpenCV's Features from Accelerated Segment Test (FAST) algorithm for
    star detection. """

    fast_alg = cv.FastFeatureDetector_create(threshold=threshold)
    # fast_alg = cv.FastFeatureDetector_create(threshold=40)
    keypoints = fast_alg.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    if not fast:
        points = prune_close_points(points, distance)

    return points


def star_tracker(star_positions_: list[tuple[int, int]],
                 detected_stars_: dict[int, dict],
                 desired_blinking_freq: float = 30,
                 fps: float = 60,
                 next_star_id: int = 0) -> tuple[dict[int, dict], int]:
    """ Function to keep track of the detected stars maintaining it's data. """
    star_positions = star_positions_.copy()
    detected_stars = detected_stars_.copy()

    for old_star_id, old_star_info in detected_stars.copy().items():
        new_star_pos = get_new_star_position(star_positions, old_star_info)

        if new_star_pos is None:
            if old_star_info["left_lifetime"] == 0:
                detected_stars.pop(old_star_id)
                continue

            position = old_star_info["position"]
            times_detected = old_star_info["times_detected"]
            left_lifetime = old_star_info["left_lifetime"] - 1

        else:
            star_positions.remove(new_star_pos)
            position = new_star_pos

            times_detected = old_star_info["times_detected"] + 1
            left_lifetime = DEFAULT_LEFT_LIFETIME

        lifetime = old_star_info["lifetime"] + 1
        blinking_freq = ((times_detected / lifetime) * fps)

        ttbts = old_star_info["tickets_to_be_the_satellite"]
        if abs(blinking_freq - desired_blinking_freq) < FREQ_THRESHOLD:
            ttbts += 1
        else:
            ttbts -= 2

        detected_stars[old_star_id].update({
            "position": position,
            "times_detected": times_detected,
            "lifetime": lifetime,
            "left_lifetime": left_lifetime,
            "blinking_freq": blinking_freq,
            "tickets_to_be_the_satellite": ttbts
        })

    detected_stars, stop_range_id = add_remaining_stars(
        star_positions, detected_stars, fps, next_star_id)

    return detected_stars, stop_range_id


def get_new_star_position(
        star_positions: list[tuple[int, int]],
        old_star_info: tuple[int, int]) -> tuple[int, int] | None:
    """ Function to get the new position of a given star. """

    old_star_pos = old_star_info["position"]

    try:
        star_positions.index(old_star_pos)
        return old_star_pos

    except ValueError:
        new_star_pos = None
        best_candidate_dist = inf

        for current_star_pos in star_positions:
            current_pair_dist = dist(old_star_pos, current_star_pos)

            if (current_pair_dist < DISTANCE
                    and best_candidate_dist > current_pair_dist):
                best_candidate_dist = current_pair_dist
                new_star_pos = current_star_pos

        return new_star_pos


def add_remaining_stars(star_positions: list[tuple[int, int]],
                        detected_stars: dict[int, dict], fps: float,
                        next_star_id: int) -> tuple[dict[int, dict], int]:
    """ Function to add the remaining stars as new ones into de stars dict. """

    stop_range_id = next_star_id + len(star_positions)
    star_ids = range(next_star_id, stop_range_id)

    for star_id, star_position in zip(star_ids, star_positions):
        detected_stars.update({
            star_id: {
                "position": star_position,
                "times_detected": 1,
                "lifetime": 1,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps,
                "tickets_to_be_the_satellite": 0,
                "color": [v * 255 for v in hsv_to_rgb(random.random(), 1, 1)],
            }
        })

    return detected_stars, stop_range_id


def detect_blinking_star(
        detected_stars: dict[int, dict]) -> tuple[int, dict] | None:
    """ Function to detect which one of the found stars is blinking the closest
    to the desired frequency. """

    return max(
        detected_stars.items(),
        key=lambda star: star[1]["tickets_to_be_the_satellite"],
        default=None,
    )
