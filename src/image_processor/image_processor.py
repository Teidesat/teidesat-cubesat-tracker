#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import cv2 as cv
import numpy as np

KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


def prune_close_points(indices, min_distance):
    """ Docstring """  # ToDo: redact docstring

    pruned = []
    for i, (y1_coord, x1_coord) in enumerate(indices):
        duplicated = False
        for y2_coord, x2_coord in indices[i + 1:]:
            current_distance = (abs(y1_coord - y2_coord) +
                                abs(x1_coord - x2_coord))
            if current_distance < min_distance:
                duplicated = True
                break
        if not duplicated:
            pruned.append((y1_coord, x1_coord))
    return pruned


def find_stars(image, threshold=20, px_sensitivity=10, fast=True, distance=20):
    """ Docstring """  # ToDo: redact docstring

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
        # Same as before but now store the original index to recover later the original coordinates
        # Then prune the positions that are too close

        indices_round = np.round(indices / px_sensitivity) * px_sensitivity
        indices_round = {(y, x): i for i, (y, x) in enumerate(indices_round)}
        indices = [(indices[i][1], indices[i][0])
                   for _, i in indices_round.items()]
        return prune_close_points(indices, distance)
