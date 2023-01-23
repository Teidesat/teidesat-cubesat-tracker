#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the Star class.
"""

from colorsys import hsv_to_rgb
from math import dist
import random
from time import time
from typing import Generator, Optional

from constants import (
    SAT_DESIRED_BLINKING_FREQ,
    VIDEO_FPS,
    DEFAULT_LEFT_LIFETIME,
    DEFAULT_VECTOR,
    MIN_HISTORY_LENGTH,
    MAX_HISTORY_LENGTH,
    REMOVE_OUTLIERS,
    MAX_OUTLIER_THRESHOLD,
    MAX_MOVE_DISTANCE,
    FREQUENCY_THRESHOLD,
)

random.seed(time())


def id_generator() -> Generator:
    """ Function to generate a new star identifier. """

    next_id = 0

    while True:
        yield next_id

        next_id += 1


class Star:
    """ Class to represent a detected star (or satellite) and save its data for
    tracking purposes. """

    _id = id_generator()

    def __init__(
            self,
            last_positions: list[tuple[int, int]] = None,
            last_times_detected: list[int] = None,
            lifetime: int = 1,
            left_lifetime: int = DEFAULT_LEFT_LIFETIME,
            blinking_freq: float = VIDEO_FPS,
            detection_confidence: int = 0,
            movement_vector: tuple[float, float] = DEFAULT_VECTOR,
            color: list[int] = None,
    ):
        self.id = next(self._id)

        self.last_positions = [
        ] if last_positions is None else last_positions
        self.last_times_detected = [
        ] if last_times_detected is None else last_times_detected

        self.lifetime = lifetime
        self.left_lifetime = left_lifetime
        self.blinking_freq = blinking_freq
        self.detection_confidence = detection_confidence
        self.movement_vector = movement_vector

        self.color = [
            v * 255 for v in hsv_to_rgb(random.random(), 1, 1)
        ] if color is None else color

    def update_info(
            self,
            star_positions: list[tuple[int, int]],
            detected_stars: dict[int],
            max_move_distance: float = MAX_MOVE_DISTANCE,
            sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
            video_fps: float = VIDEO_FPS,
            default_left_lifetime: int = DEFAULT_LEFT_LIFETIME,
            max_history_length: int = MAX_HISTORY_LENGTH,
            min_history_length: int = MIN_HISTORY_LENGTH,
            remove_outliers: bool = REMOVE_OUTLIERS,
            max_outlier_threshold: float = MAX_OUTLIER_THRESHOLD,
            default_vector: tuple[float, float] = DEFAULT_VECTOR,
            frequency_threshold: float = FREQUENCY_THRESHOLD,
    ) -> None:
        """ Function to update the star's information.

        Note: This function modifies data from 'star_positions' and
        'detected_stars' parameters without an explicit return statement for
        memory usage reduction purposes.
        """

        new_star_pos = self.get_new_star_position(star_positions,
                                                  max_move_distance)

        if new_star_pos is None:
            if self.left_lifetime == 0:
                detected_stars.pop(self.id)
                return

            self.last_times_detected.append(0)
            self.left_lifetime -= 1

        else:
            star_positions.remove(new_star_pos)
            self.last_positions.append(new_star_pos)
            self.last_times_detected.append(1)
            self.left_lifetime = default_left_lifetime

        self.last_positions = \
            self.last_positions[-max_history_length:]
        self.last_times_detected = \
            self.last_times_detected[-max_history_length:]

        self.lifetime += 1

        self.movement_vector = self.get_movement_vector(min_history_length,
                                                        remove_outliers,
                                                        max_outlier_threshold,
                                                        default_vector)

        self.blinking_freq = video_fps * (sum(self.last_times_detected) /
                                          len(self.last_times_detected))

        self.detection_confidence += 1 if (
                abs(self.blinking_freq -
                    sat_desired_blinking_freq) < frequency_threshold
        ) else -2

    def get_new_star_position(
            self,
            star_positions: list[tuple[int, int]],
            max_move_distance: float = MAX_MOVE_DISTANCE,
    ) -> Optional[tuple[int, int]]:
        """ Function to get the new position of the star. """

        if len(star_positions) == 0:
            return None

        expected_star_pos = (
            round(self.last_positions[-1][0] + self.movement_vector[0]),
            round(self.last_positions[-1][1] + self.movement_vector[1])
        )

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
            self,
            min_history_length: int = MIN_HISTORY_LENGTH,
            remove_outliers: bool = REMOVE_OUTLIERS,
            max_outlier_threshold: float = MAX_OUTLIER_THRESHOLD,
            default_vector: tuple[float, float] = DEFAULT_VECTOR,
    ) -> tuple[float, float]:
        """ Function to calculate the star movement vector based on it's last
        positions. """

        if len(self.last_positions) < min_history_length:
            return default_vector

        movement_vectors = [
            (pos_2[0] - pos_1[0], pos_2[1] - pos_1[1])
            for pos_1, pos_2 in zip(self.last_positions,
                                    self.last_positions[1:])
        ]

        mean_vector = get_mean_vector(movement_vectors, default_vector)

        if not remove_outliers:
            return mean_vector

        filtered_vectors = [
            current_vector for current_vector in movement_vectors
            if dist(mean_vector, current_vector) < max_outlier_threshold
        ]

        return get_mean_vector(filtered_vectors, default_vector)


def get_mean_vector(
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
