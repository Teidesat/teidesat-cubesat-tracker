#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions and star
detection and tracking algorithms.
"""

from colorsys import hsv_to_rgb
from math import inf, dist
import random
from time import time
from typing import Union, Optional

import cv2 as cv
import numpy as np

# * Constants
from src.utils import Point, Star

THRESHOLD = 50
FAST = True
DISTANCE = 20

SAT_DESIRED_BLINKING_FREQ = 30
VIDEO_FPS = 60
MOVEMENT_THRESHOLD = 2

DEFAULT_LEFT_LIFETIME = 10
DEFAULT_MOVEMENT_VECTOR = (0, 0)

MIN_HISTORY_LEN = 10
MAX_HISTORY_LEN = 20

REMOVE_OUTLIERS = True
MAX_OUTLIER_THRESHOLD = 1.5
MAX_MOVE_DISTANCE = 10

FREQUENCY_THRESHOLD = 3
MIN_DETECTION_CONFIDENCE = 20

random.seed(time())


def prune_close_points(points: list[Point], min_dist_sq: float) -> list[Point]:
    """ Prune close points since they have a high probability of being an image artifact of the same star """
    return [
        p1 for i, p1 in enumerate(points) if
        all(p1.sq_distance(p2) > min_dist_sq for p2 in points[i+1:])
    ]


fast_detector = cv.FastFeatureDetector_create(threshold=50)


def find_stars(image, min_dist_sq=1., fast=True) -> list[Point]:
    """ Function to get all the bright points of a given image. """

    points = [Point(x=keypoint.pt[0], y=keypoint.pt[1]) for keypoint in fast_detector.detect(image, None)]
    return points if fast else prune_close_points(points, min_dist_sq)


def track_stars(star_positions: list[Point], detected_stars: dict[int, Star], video_fps: float, desired_blinking_freq: float):
    """ Keep track of the detected stars maintaining its data. """

    # Correlate new detections with previous detected stars
    new_detected_stars = {}
    for star_id, star in detected_stars.items():
        detected_pos = get_new_star_position(star_positions, star)
        star.add_detection(video_fps, desired_blinking_freq, detected_pos)
        if star.lifetime > 0:
            # TODO add expected position to history anyway if we dont find the star ?
            new_detected_stars[star_id] = star
            if detected_pos:
                star_positions.remove(detected_pos)

    # Add new stars found
    for star_position in star_positions:
        star = Star()
        star.add_detection(video_fps, desired_blinking_freq, star_position)
        detected_stars[star.star_id] = star


def get_new_star_position(star_positions: list[Point], star_info: Star, max_move_sq_distance=100.) -> Optional[tuple[Point]]:
    """ Find the closest point in range to the expected star position for a list of possible candidate positions. """
    expected_pos = star_info.expected_pos()
    star_pos = None
    smallest_sq_dist = inf

    for point in star_positions:
        point_sq_dist = point.sq_distance(expected_pos)

        if point_sq_dist < smallest_sq_dist:
            smallest_sq_dist = point_sq_dist
            star_pos = point

    return None if (smallest_sq_dist > max_move_sq_distance) else star_pos


def get_speed_vec(pos_history: list[Point], remove_outliers=True):
    """ Calculate the star movement vector. """

    if len(pos_history) < MIN_HISTORY_LEN:
        return Point(0, 0)

    # Get speed vectors average
    speed_vecs = [pos_history[i + 1] - pos_history[i] for i in range(len(pos_history) - 1)]
    mean_speed = Point(
            x=np.average(speed_vec.x for speed_vec in speed_vecs),
            y=np.average(speed_vec.y for speed_vec in speed_vecs))

    if not remove_outliers:
        return mean_speed

    # If we want to remove outliers, remove those that are too far from the average
    speed_vecs = [
        speed_vec for speed_vec in speed_vecs if mean_speed.sq_distance(speed_vec) < MAX_OUTLIER_THRESHOLD
    ]

    # If we do still have some points, return the average of those
    if speed_vecs:
        return Point(
            x=np.average(speed_vec.x for speed_vec in speed_vecs),
            y=np.average(speed_vec.y for speed_vec in speed_vecs))

    return Point(0, 0)


def detect_shooting_stars(detected_stars: dict[int, Star], sq_movement_threshold: float = 2 ** 2) -> dict[int, Star]:
    """ Detect which of the found stars are shooting stars or satellites. """
    return {k: v for k, v in detected_stars.items() if v.speed_vec.sq_distance() >= sq_movement_threshold}


def detect_blinking_star(detected_stars: dict[int, Star], min_detect_confidence=20) -> Optional[Star]:
    """ Detect which one of the found stars is blinking the closest
    to the desired frequency. """

    blinking_star: Star = max(
        detected_stars.values(),
        key=lambda star: star.detection_confidence,
        default=None,
    )

    return blinking_star \
        if (blinking_star is not None and blinking_star.detection_confidence > min_detect_confidence) else None
