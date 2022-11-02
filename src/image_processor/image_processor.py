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
from typing import Optional

import numpy as np

# * Constants
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

new_star_id = 0


def prune_close_points(points: list[tuple[int, int]],
                       min_distance: float) -> list[tuple[int, int]]:
    """ Prune close points since they have a high probability of being an image
    artifact of the same star """

    return [
        point_1 for i, point_1 in enumerate(points) if
        all(dist(point_1, point_2) > min_distance
            for point_2 in points[i + 1:])
    ]


def detect_stars(image,
                 star_detector,
                 fast: bool = FAST,
                 distance: float = DISTANCE) -> list[tuple[int, int]]:
    """ Function to get all the bright points of a given image. """

    keypoints = star_detector.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    return points if fast else prune_close_points(points, distance)


def track_stars(star_positions: list[tuple[int, int]],
                detected_stars: dict[int, dict],
                desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
                video_fps: float = VIDEO_FPS) -> None:
    """ Function to keep track of the detected stars maintaining its data.

    Note: This function modifies data from 'star_positions' and
    'detected_stars' parameters without an explicit return statement for memory
    usage reduction purposes.
    """

    for old_star in detected_stars.copy().items():
        update_star_info(
            old_star,
            star_positions,
            detected_stars,
            desired_blinking_freq,
            video_fps,
        )

    add_remaining_stars(
        star_positions,
        detected_stars,
        video_fps,
    )


def add_remaining_stars(star_positions: list[tuple[int, int]],
                        detected_stars: dict[int, dict],
                        video_fps: float) -> None:
    """ Function to add the remaining stars as new ones into de stars dict.

    Note: This function modifies data from 'detected_stars' parameter without
    an explicit return statement for memory usage reduction purposes.
    """

    global new_star_id

    for star_position in star_positions:
        detected_stars[new_star_id] = {
            "last_positions": [star_position],
            "last_times_detected": [1],
            "lifetime": 1,
            "left_lifetime": DEFAULT_LEFT_LIFETIME,
            "blinking_freq": video_fps,
            "detection_confidence": 0,
            "movement_vector": DEFAULT_MOVEMENT_VECTOR,
            "color": [v * 255 for v in hsv_to_rgb(random.random(), 1, 1)],
        }

        new_star_id += 1


def update_star_info(old_star: tuple[int, dict],
                     star_positions: list[tuple[int, int]],
                     detected_stars: dict[int, dict],
                     desired_blinking_freq: float,
                     video_fps: float) -> None:
    """ Function to update a star's information.

    Note: This function modifies data from 'star_positions' and
    'detected_stars' parameters without an explicit return statement for memory
    usage reduction purposes.
    """

    old_star_id, old_star_info = old_star

    new_star_pos = get_new_star_position(star_positions, old_star_info)

    if new_star_pos is None:
        if old_star_info["left_lifetime"] == 0:
            detected_stars.pop(old_star_id)
            return

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
    blinking_freq = video_fps * (sum(last_times_detected) /
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


def get_new_star_position(star_positions: list[tuple[int, int]],
                          old_star_info: dict) -> Optional[tuple[int, int]]:
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


def get_movement_vector(last_positions: list[tuple[int, int]]):
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


def detect_shooting_stars(
        detected_stars: dict[int, dict],
        movement_threshold: float = MOVEMENT_THRESHOLD) -> dict[int, dict]:
    """ Function to detect which of the found stars are shooting stars or
    satellites. """

    return dict(
        filter(
            lambda star:
            (np.linalg.norm(star[1]["movement_vector"]) >= movement_threshold),
            detected_stars.items()))


def detect_blinking_star(
        detected_stars: dict[int, dict]) -> Optional[tuple[int, dict]]:
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
