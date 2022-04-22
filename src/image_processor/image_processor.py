#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions and star
detection algorithms.
"""

from math import inf, dist

import cv2 as cv
import numpy as np
import skimage.feature

#* Constants
THRESHOLD = 40
PX_SENSITIVITY = 8
FAST = True
DISTANCE = 20
BEST_ALGORITHM_INDEX = 8
DEFAULT_LIFETIME = 20

KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


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


def star_tracker(star_positions: list[tuple[int, int]], detected_stars: dict,
                 processed_frames: int, fps: float) -> dict:
    """ Function to keep track of the detected stars maintaining it's data. """

    for old_star_pos, star_info in detected_stars.copy().items():
        equivalent_star = None
        best_candidate_dist = inf
        for new_star_pos in star_positions:
            current_pair_dist = dist(old_star_pos, new_star_pos)

            if (current_pair_dist < DISTANCE
                    and best_candidate_dist > current_pair_dist):
                best_candidate_dist = current_pair_dist
                equivalent_star = new_star_pos

        if equivalent_star is not None:
            star_positions.remove(equivalent_star)
            detected_stars.pop(old_star_pos)
            detected_stars.update({
                equivalent_star: {
                    "times_detected":
                    star_info["times_detected"] + 1,
                    "lifetime":
                    DEFAULT_LIFETIME,
                    "blinking_freq":
                    ((star_info["times_detected"] / processed_frames) * fps),
                }
            })

        else:
            if star_info["lifetime"] == 0:
                detected_stars.pop(old_star_pos)
            else:
                detected_stars.update({
                    old_star_pos: {
                        "times_detected":
                        star_info["times_detected"],
                        "lifetime":
                        star_info["lifetime"] - 1,
                        "blinking_freq":
                        ((star_info["times_detected"] / processed_frames) *
                         fps),
                    }
                })

    detected_stars.update(
        dict.fromkeys(
            star_positions, {
                "times_detected": 1,
                "lifetime": DEFAULT_LIFETIME,
                "blinking_freq": fps,
            }))

    return detected_stars


def detect_blinking_star(detected_stars: dict,
                         desired_blinking_freq: float) -> tuple[int, int]:
    """ Function to detect which one of the found stars is blinking the closest
    to the desired frequency. """

    return min(
        detected_stars.items(),
        key=lambda star: abs(star[1]["blinking_freq"] - desired_blinking_freq),
        default=None,
    )
