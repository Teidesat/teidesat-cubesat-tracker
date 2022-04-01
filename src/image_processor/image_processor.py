#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions and star
detection algorithms.
"""

import cv2 as cv
import numpy as np
import skimage.feature

KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


def prune_close_points(indices: list[tuple[int, int]],
                       min_distance: float) -> list[tuple[int, int]]:
    """ Function to prune close points because of their probabilities of being a
    duplicate reference to the same element of an image. """

    pruned = []
    for i, (y1_coord, x1_coord) in enumerate(indices):
        duplicate = False
        for y2_coord, x2_coord in indices[i + 1:]:
            distance_between_current_points = (abs(y1_coord - y2_coord) +
                                               abs(x1_coord - x2_coord))
            if distance_between_current_points < min_distance:
                duplicate = True
                break
        if not duplicate:
            pruned.append((y1_coord, x1_coord))
    return pruned


def find_stars(image,
               threshold: float = 20,
               px_sensitivity: int = 10,
               fast: bool = True,
               distance: float = 20,
               algorithm_index: int = 8) -> list[tuple[int, int]]:
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
        jorge_algorithm,
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


def jorge_algorithm(image, threshold: float, px_sensitivity: int, fast: bool,
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

    fast_alg = cv.FastFeatureDetector_create(threshold=40)
    keypoints = fast_alg.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    if not fast:
        points = prune_close_points(list(points), distance)

    return points
