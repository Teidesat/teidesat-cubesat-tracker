#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions, star detection and star
 tracking algorithms.
"""

from math import dist
from typing import Optional

from src.star import Star

from constants import (
    VIDEO_FPS,
    PRUNE_CLOSE_POINTS,
    MIN_PRUNE_DISTANCE,
    DEFAULT_LEFT_LIFETIME,
    SAT_DESIRED_BLINKING_FREQ,
    MIN_DETECTION_CONFIDENCE,
    DEFAULT_VECTOR,
    MOVEMENT_THRESHOLD,
)


def detect_stars(
    image,
    star_detector,
    prune_close_points: bool = PRUNE_CLOSE_POINTS,
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
) -> list[tuple[int, int]]:
    """Function to get all the bright points of a given image."""

    keypoints = star_detector.detect(image, None)
    points = [keypoint.pt for keypoint in keypoints]

    return (
        _prune_close_points(points, min_prune_distance)
        if prune_close_points
        else points
    )


def _prune_close_points(
    points: list[tuple[int, int]],
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
) -> list[tuple[int, int]]:
    """Prune close points since they have a high probability of being an image artifact
    of the same star."""

    return [
        point_1
        for i, point_1 in enumerate(points)
        if all(
            dist(point_1, point_2) > min_prune_distance for point_2 in points[(i + 1) :]
        )
    ]


def track_stars(
    star_positions: list[tuple[int, int]],
    detected_stars: set[Star],
    sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
    video_fps: float = VIDEO_FPS,
    default_left_lifetime: int = DEFAULT_LEFT_LIFETIME,
    default_vector: tuple[float, float] = DEFAULT_VECTOR,
) -> None:
    """
    Function to keep track of the detected stars maintaining its data.
    <br/><br/>

    Note: This function modifies data from 'star_positions' and 'detected_stars'
    parameters without an explicit return statement for memory usage reduction purposes.
    """

    # Update previously detected stars
    for old_star in detected_stars.copy():
        old_star.update_info(
            star_positions,
            detected_stars,
            sat_desired_blinking_freq=sat_desired_blinking_freq,
            video_fps=video_fps,
            default_left_lifetime=default_left_lifetime,
            default_vector=default_vector,
        )

    # Add newly detected stars
    for star_position in star_positions:
        detected_stars.add(
            Star(
                last_positions=[star_position],
                last_times_detected=[1],
                lifetime=1,
                left_lifetime=default_left_lifetime,
                blinking_freq=video_fps,
                detection_confidence=0,
                movement_vector=default_vector,
            )
        )


def detect_shooting_stars(
    detected_stars: set[Star],
    movement_threshold: float = MOVEMENT_THRESHOLD,
) -> set[Star]:
    """Function to detect which of the found stars are shooting stars or satellites."""

    return {
        star
        for star in detected_stars
        if (abs(star.movement_vector[0]) + abs(star.movement_vector[1]))
        >= movement_threshold
    }


def detect_blinking_star(
    detected_stars: set[Star],
    min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
) -> Optional[Star]:
    """Function to detect which one of the found stars has the highest confidence of
    being the satellite."""

    blinking_star = max(
        detected_stars,
        key=lambda star: star.detection_confidence,
        default=None,
    )

    if (
        blinking_star is not None
        and blinking_star.detection_confidence > min_detection_confidence
    ):
        return blinking_star

    return None
