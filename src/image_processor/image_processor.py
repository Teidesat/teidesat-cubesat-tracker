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
DEFAULT_MOVEMENT_VECTOR = (0, 0)

MIN_HISTORY_LEN = 10
MAX_HISTORY_LEN = 20

REMOVE_OUTLIERS = True
MAX_OUTLIER_THRESHOLD = 1.5
MAX_MOVE_DISTANCE = 10

FREQUENCY_THRESHOLD = 3
MIN_DETECTION_CONFIDENCE = 20

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

            old_star_info["last_times_detected"].append(0)
            left_lifetime = old_star_info["left_lifetime"] - 1

        else:
            star_positions.remove(new_star_pos)
            old_star_info["last_positions"].append(new_star_pos)

            old_star_info["last_times_detected"].append(1)
            left_lifetime = DEFAULT_LEFT_LIFETIME

        last_positions = old_star_info["last_positions"][-MAX_HISTORY_LEN:]
        last_times_detected = old_star_info["last_times_detected"][
            -MAX_HISTORY_LEN:]
        movement_vector = get_movement_vector(last_positions)

        lifetime = old_star_info["lifetime"] + 1
        blinking_freq = fps * (sum(last_times_detected) /
                               len(last_times_detected))

        detection_confidence = old_star_info["detection_confidence"]
        if abs(blinking_freq - desired_blinking_freq) < FREQUENCY_THRESHOLD:
            detection_confidence += 1
        else:
            detection_confidence -= 2

        detected_stars[old_star_id].update({
            "last_positions": last_positions,
            "last_times_detected": last_times_detected,
            "lifetime": lifetime,
            "left_lifetime": left_lifetime,
            "blinking_freq": blinking_freq,
            "detection_confidence": detection_confidence,
            "movement_vector": movement_vector,
        })

    detected_stars, stop_range_id = add_remaining_stars(
        star_positions, detected_stars, fps, next_star_id)

    return detected_stars, stop_range_id


def get_new_star_position(
        star_positions: list[tuple[int, int]],
        old_star_info: tuple[int, int]) -> tuple[int, int] | None:
    """ Function to get the new position of a given star. """

    expected_star_pos = (old_star_info["last_positions"][-1][0] +
                         old_star_info["movement_vector"][0],
                         old_star_info["last_positions"][-1][1] +
                         old_star_info["movement_vector"][1])

    try:
        star_positions.index(expected_star_pos)
        return expected_star_pos

    except ValueError:
        new_star_pos = None
        best_candidate_dist = inf

        for current_star_pos in star_positions:
            current_pair_dist = dist(expected_star_pos, current_star_pos)

            if (current_pair_dist < MAX_MOVE_DISTANCE
                    and best_candidate_dist > current_pair_dist):
                best_candidate_dist = current_pair_dist
                new_star_pos = current_star_pos

        return new_star_pos


def get_movement_vector(last_positions):
    """ Function to calculate the star movement vector. """

    if len(last_positions) < MIN_HISTORY_LEN:
        return DEFAULT_MOVEMENT_VECTOR

    movement_vectors = [
        (point_2[0] - point_1[0], point_2[1] - point_1[1])
        for point_1, point_2 in zip(last_positions, last_positions[1:])
    ]

    m_v_len = len(movement_vectors)
    mean_vector = [sum(values) / m_v_len for values in zip(*movement_vectors)]

    if not REMOVE_OUTLIERS:
        return mean_vector

    filtered_vectors = [
        current_vector for current_vector in movement_vectors
        if dist(mean_vector, current_vector) < MAX_OUTLIER_THRESHOLD
    ]

    f_v_len = len(filtered_vectors)
    if f_v_len != 0:
        return [sum(values) / f_v_len for values in zip(*filtered_vectors)]

    return DEFAULT_MOVEMENT_VECTOR


def add_remaining_stars(star_positions: list[tuple[int, int]],
                        detected_stars: dict[int, dict], fps: float,
                        next_star_id: int) -> tuple[dict[int, dict], int]:
    """ Function to add the remaining stars as new ones into de stars dict. """

    stop_range_id = next_star_id + len(star_positions)
    star_ids = range(next_star_id, stop_range_id)

    for star_id, star_position in zip(star_ids, star_positions):
        detected_stars.update({
            star_id: {
                "last_positions": [star_position],
                "last_times_detected": [1],
                "lifetime": 1,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps,
                "detection_confidence": 0,
                "movement_vector": DEFAULT_MOVEMENT_VECTOR,
                "color": [v * 255 for v in hsv_to_rgb(random.random(), 1, 1)],
            }
        })

    return detected_stars, stop_range_id


def detect_shooting_stars(detected_stars: dict[int, dict],
                          movement_threshold: float = 2) -> dict[int, dict]:
    """ Function to detect which of the found stars are shooting stars or
    satellites. """

    return dict(
        filter(
            lambda star:
            (np.linalg.norm(star[1]["movement_vector"]) >= movement_threshold),
            detected_stars.items()))


def detect_blinking_star(
        detected_stars: dict[int, dict]) -> tuple[int, dict] | None:
    """ Function to detect which one of the found stars is blinking the closest
    to the desired frequency. """

    blinking_star = max(
        detected_stars.items(),
        key=lambda star: star[1]["detection_confidence"],
        default=None,
    )

    if ((blinking_star is not None) and
        (blinking_star[1]["detection_confidence"] > MIN_DETECTION_CONFIDENCE)):
        return blinking_star
    return None
