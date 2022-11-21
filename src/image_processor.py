#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions, star detection
 and star tracking algorithms.
"""

from colorsys import hsv_to_rgb
from math import dist
import random
from time import time
from typing import Optional

# * Constants
FAST = True
MIN_PRUNE_DISTANCE = 20.0

SAT_DESIRED_BLINKING_FREQ = 30.0
VIDEO_FPS = 60
MOVEMENT_THRESHOLD = 3.0

DEFAULT_LEFT_LIFETIME = 10
DEFAULT_VECTOR = (0.0, 0.0)

MIN_HISTORY_LENGTH = 10
MAX_HISTORY_LENGTH = 20

REMOVE_OUTLIERS = True
MAX_OUTLIER_THRESHOLD = 1.5
MAX_MOVE_DISTANCE = 10.0

FREQUENCY_THRESHOLD = 3.0
MIN_DETECTION_CONFIDENCE = 20

random.seed(time())

new_star_id = 0


def prune_close_points(
        points: list[tuple[int, int]],
        min_prune_distance: float = MIN_PRUNE_DISTANCE,
) -> list[tuple[int, int]]:
    """ Prune close points since they have a high probability of being an image
    artifact of the same star """

    return [
        point_1 for i, point_1 in enumerate(points) if
        all(dist(point_1, point_2) > min_prune_distance
            for point_2 in points[i + 1:])
    ]


def detect_stars(
        image,
        star_detector,
        fast: bool = FAST,
        min_prune_distance: float = MIN_PRUNE_DISTANCE,
) -> list[tuple[int, int]]:
    """ Function to get all the bright points of a given image. """

    keypoints = star_detector.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    return points if fast else prune_close_points(points, min_prune_distance)


def track_stars(
        star_positions: list[tuple[int, int]],
        detected_stars: dict[int, dict],
        sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
        video_fps: float = VIDEO_FPS,
) -> None:
    """ Function to keep track of the detected stars maintaining its data.

    Note: This function modifies data from 'star_positions' and
    'detected_stars' parameters without an explicit return statement for memory
    usage reduction purposes.
    """

    for old_star_id in list(detected_stars.keys()):
        update_star_info(
            old_star_id,
            star_positions,
            detected_stars,
            sat_desired_blinking_freq,
            video_fps,
        )

    add_remaining_stars(
        star_positions,
        detected_stars,
        video_fps,
    )


def add_remaining_stars(
        star_positions: list[tuple[int, int]],
        detected_stars: dict[int, dict],
        video_fps: float = VIDEO_FPS,
        default_left_lifetime: int = DEFAULT_LEFT_LIFETIME,
        default_movement_vector: tuple[float, float] = DEFAULT_VECTOR,
) -> None:
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
            "left_lifetime": default_left_lifetime,
            "blinking_freq": video_fps,
            "detection_confidence": 0,
            "movement_vector": default_movement_vector,
            "color": [v * 255 for v in hsv_to_rgb(random.random(), 1, 1)],
        }

        new_star_id += 1


def update_star_info(
        old_star_id: int,
        star_positions: list[tuple[int, int]],
        detected_stars: dict[int, dict],
        sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
        video_fps: float = VIDEO_FPS,
        default_left_lifetime: int = DEFAULT_LEFT_LIFETIME,
        max_history_length: int = MAX_HISTORY_LENGTH,
        frequency_threshold: float = FREQUENCY_THRESHOLD,
) -> None:
    """ Function to update a star's information.

    Note: This function modifies data from 'star_positions' and
    'detected_stars' parameters without an explicit return statement for memory
    usage reduction purposes.
    """

    old_star_info = detected_stars[old_star_id]

    new_star_pos = get_new_star_position(star_positions, old_star_info)

    if new_star_pos is None:
        if old_star_info["left_lifetime"] == 0:
            detected_stars.pop(old_star_id)
            return

        old_star_info["last_times_detected"].append(0)
        old_star_info["left_lifetime"] -= 1

    else:
        star_positions.remove(new_star_pos)
        old_star_info["last_positions"].append(new_star_pos)
        old_star_info["last_times_detected"].append(1)
        old_star_info["left_lifetime"] = default_left_lifetime

    old_star_info["last_positions"] = \
        old_star_info["last_positions"][-max_history_length:]

    old_star_info["last_times_detected"] = \
        old_star_info["last_times_detected"][-max_history_length:]

    old_star_info["lifetime"] += 1

    old_star_info["movement_vector"] = get_movement_vector(
        old_star_info["last_positions"]
    )

    old_star_info["blinking_freq"] = video_fps * (
        sum(old_star_info["last_times_detected"]) /
        len(old_star_info["last_times_detected"])
    )

    old_star_info["detection_confidence"] += 1 if (
            abs(old_star_info["blinking_freq"] - sat_desired_blinking_freq) <
            frequency_threshold
    ) else -2


def get_new_star_position(
        star_positions: list[tuple[int, int]],
        old_star_info: dict,
        max_move_distance: float = MAX_MOVE_DISTANCE,
) -> Optional[tuple[int, int]]:
    """ Function to get the new position of a given star. """

    expected_star_pos = (round(old_star_info["last_positions"][-1][0] +
                               old_star_info["movement_vector"][0]),
                         round(old_star_info["last_positions"][-1][1] +
                               old_star_info["movement_vector"][1]))

    try:
        star_positions.index(expected_star_pos)
        return expected_star_pos

    except ValueError:
        new_star_pos = None
        best_candidate_dist = max_move_distance

        for current_star_pos in star_positions:
            current_pair_dist = dist(expected_star_pos, current_star_pos)

            if current_pair_dist < best_candidate_dist:
                best_candidate_dist = current_pair_dist
                new_star_pos = current_star_pos

        return new_star_pos


def get_movement_vector(
        last_positions: list[tuple[int, int]],
        min_history_length: int = MIN_HISTORY_LENGTH,
        remove_outliers: bool = REMOVE_OUTLIERS,
        max_outlier_threshold: float = MAX_OUTLIER_THRESHOLD,
        default_vector: tuple[float, float] = DEFAULT_VECTOR,
) -> tuple[float, float]:
    """ Function to calculate the star movement vector based on it's last
    positions. """

    if len(last_positions) < min_history_length:
        return default_vector

    movement_vectors = [
        (point_2[0] - point_1[0], point_2[1] - point_1[1])
        for point_1, point_2 in zip(last_positions, last_positions[1:])
    ]

    mean_vector = get_mean_vect(movement_vectors, default_vector)

    if not remove_outliers:
        return mean_vector

    filtered_vectors = [
        current_vector for current_vector in movement_vectors
        if dist(mean_vector, current_vector) < max_outlier_threshold
    ]

    return get_mean_vect(filtered_vectors, default_vector)


def get_mean_vect(
        points: list[tuple[int, int]],
        default_vector: tuple[float, float] = DEFAULT_VECTOR,
) -> tuple[float, float]:
    """ Function to calculate the mean vector of the given list of points. """

    num_of_points = len(points)

    if num_of_points != 0:
        zipped_points = list(zip(*points))

        return (sum(zipped_points[0]) / num_of_points,
                sum(zipped_points[1]) / num_of_points)

    return default_vector


def detect_shooting_stars(
        detected_stars: dict[int, dict],
        movement_threshold: float = MOVEMENT_THRESHOLD,
) -> dict[int, dict]:
    """ Function to detect which of the found stars are shooting stars or
    satellites. """

    return {
        star_id: star_info
        for star_id, star_info in detected_stars.items()
        if (abs(star_info["movement_vector"][0]) +
            abs(star_info["movement_vector"][1])) >= movement_threshold
    }


def detect_blinking_star(
        detected_stars: dict[int, dict],
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
) -> Optional[tuple[int, dict]]:
    """ Function to detect which one of the found stars has the highest
    confidence of being the satellite. """

    blinking_star = max(
        detected_stars.items(),
        key=lambda star: star[1]["detection_confidence"],
        default=None,
    )

    assert blinking_star is None or isinstance(blinking_star, tuple)

    if (blinking_star is not None and
            (blinking_star[1]["detection_confidence"] >
             min_detection_confidence)):
        return blinking_star

    return None
