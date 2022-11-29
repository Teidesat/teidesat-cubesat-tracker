#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions, star detection
 and star tracking algorithms.
"""

from math import dist
from typing import Optional

from src.star import Star

# * Constants
FAST = True
MIN_PRUNE_DISTANCE = 20.0

SAT_DESIRED_BLINKING_FREQ = 30.0
VIDEO_FPS = 60.0

DEFAULT_LEFT_LIFETIME = 10
DEFAULT_VECTOR = (0.0, 0.0)

MOVEMENT_THRESHOLD = 3.0
MIN_DETECTION_CONFIDENCE = 20


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


def track_stars(
        star_positions: list[tuple[int, int]],
        detected_stars: dict[int, Star],
        sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
        video_fps: float = VIDEO_FPS,
        default_left_lifetime: int = DEFAULT_LEFT_LIFETIME,
        default_vector: tuple[float, float] = DEFAULT_VECTOR,
) -> None:
    """ Function to keep track of the detected stars maintaining its data.

    Note: This function modifies data from 'star_positions' and
    'detected_stars' parameters without an explicit return statement for memory
    usage reduction purposes.
    """

    # Update previously detected stars
    for old_star_id in list(detected_stars.keys()):
        detected_stars[old_star_id].update_info(
            star_positions,
            detected_stars,
            sat_desired_blinking_freq=sat_desired_blinking_freq,
            video_fps=video_fps,
            default_left_lifetime=default_left_lifetime,
            default_vector=default_vector,
        )

    # Add newly detected stars
    for star_position in star_positions:
        new_star = Star(
           last_positions=[star_position],
           last_times_detected=[1],
           lifetime=1,
           left_lifetime=default_left_lifetime,
           blinking_freq=video_fps,
           detection_confidence=0,
           movement_vector=default_vector,
        )

        detected_stars[new_star.id] = new_star


def detect_shooting_stars(
        detected_stars: dict[int, Star],
        movement_threshold: float = MOVEMENT_THRESHOLD,
) -> dict[int, Star]:
    """ Function to detect which of the found stars are shooting stars or
    satellites. """

    return {
        star_id: star_info
        for star_id, star_info in detected_stars.items()
        if (abs(star_info.movement_vector[0]) +
            abs(star_info.movement_vector[1])) >= movement_threshold
    }


def detect_blinking_star(
        detected_stars: dict[int, Star],
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
) -> Optional[tuple[int, Star]]:
    """ Function to detect which one of the found stars has the highest
    confidence of being the satellite. """

    blinking_star = max(
        detected_stars.items(),
        key=lambda star: star[1].detection_confidence,
        default=None,
    )

    assert blinking_star is None or isinstance(blinking_star, tuple)

    if (blinking_star is not None and
            (blinking_star[1].detection_confidence >
             min_detection_confidence)):
        return blinking_star

    return None
