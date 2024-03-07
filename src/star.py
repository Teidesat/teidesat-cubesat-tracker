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
    MOVEMENT_VECTOR_COMPUTATION_METHOD,
)

random.seed(time())


def id_generator() -> Generator:
    """Function to generate unique star identifiers."""

    next_id = 0

    while True:
        yield next_id

        next_id += 1


# @dataclass  # ToDo: test if implementing dataclass does not lead to a loss of performance
class Star:
    """Class to represent a detected star (or satellite) and save its data for tracking
    purposes."""

    _id = id_generator()

    def __init__(
        self,
        last_positions: list[Optional[tuple[int, int]]] = None,
        last_times_detected: list[int] = None,
        lifetime: int = 1,
        left_lifetime: int = DEFAULT_LEFT_LIFETIME,
        blinking_freq: float = VIDEO_FPS,
        detection_confidence: int = 0,
        movement_vector: tuple[float, float] = DEFAULT_VECTOR,
        color: list[int] = None,
        frames_since_last_detection: int = None,
        last_detected_position: tuple[int, int] = None,
        expected_position: tuple[int, int] = None,
    ):
        self.id = next(self._id)

        self.last_positions = [] if last_positions is None else last_positions
        self.last_times_detected = (
            [] if last_times_detected is None else last_times_detected
        )

        self.lifetime = lifetime
        self.left_lifetime = left_lifetime
        self.blinking_freq = blinking_freq
        self.detection_confidence = detection_confidence
        self.movement_vector = movement_vector

        self.color = (
            [v * 255 for v in hsv_to_rgb(random.random(), 1, 1)]
            if color is None
            else color
        )

        self.frames_since_last_detection = None
        self.last_detected_position = self.get_new_last_detected_position(
            last_detected_position
        )

        if frames_since_last_detection is not None:
            self.frames_since_last_detection = frames_since_last_detection

        self.expected_position = self.get_new_expected_position(expected_position)

    def __hash__(self) -> int:
        """Function to set the star's id as the object's hash value."""
        return self.id

    def __eq__(self, other) -> bool:
        """Function to compare two stars by their attributes."""
        return isinstance(other, Star) and self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        """Function to add the star attributes to the default representation."""
        return super().__repr__() + ": " + str(self.__dict__)

    def update_info(
        self,
        star_positions: list[tuple[int, int]],
        detected_stars: set,  # set[Star]
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
        movement_vector_computation_method: str = MOVEMENT_VECTOR_COMPUTATION_METHOD,
    ) -> None:
        """
        Function to update the star's information.
        <br/><br/>

        Note: This function modifies data from 'star_positions' and 'detected_stars'
        parameters without an explicit return statement for memory usage reduction
        purposes.
        """

        new_star_pos = self.get_new_star_position(star_positions, max_move_distance)

        if new_star_pos is None:
            if self.left_lifetime == 0:
                detected_stars.remove(self)
                return

            self.last_positions.append(None)
            self.last_times_detected.append(0)
            self.left_lifetime -= 1
            self.frames_since_last_detection += 1

        else:
            star_positions.remove(new_star_pos)
            self.last_positions.append(new_star_pos)
            self.last_times_detected.append(1)
            self.left_lifetime = default_left_lifetime
            self.frames_since_last_detection = 1
            self.last_detected_position = new_star_pos

        self.last_positions = self.last_positions[-max_history_length:]
        self.last_times_detected = self.last_times_detected[-max_history_length:]

        self.lifetime += 1

        self.movement_vector = self.get_new_movement_vector(
            min_history_length,
            remove_outliers,
            max_outlier_threshold,
            default_vector,
            movement_vector_computation_method,
        )
        self.expected_position = self.get_new_expected_position()

        self.blinking_freq = video_fps * (
            sum(self.last_times_detected) / len(self.last_times_detected)
        )

        self.detection_confidence += (
            1
            if (
                abs(self.blinking_freq - sat_desired_blinking_freq)
                < frequency_threshold
            )
            else -2
        )

    def get_new_star_position(
        self,
        star_positions: list[tuple[int, int]],
        max_move_distance: float = MAX_MOVE_DISTANCE,
    ) -> Optional[tuple[int, int]]:
        """
        Function to get the new position of the star.
        <br/><br/>

        Note: If the prediction is not pixel perfect then the closest star position is
        returned.
        """

        if len(star_positions) == 0:
            return None

        try:
            star_positions.index(self.expected_position)
            return self.expected_position

        except ValueError:
            new_star_pos = None
            best_candidate_dist = max_move_distance

            for current_star_pos in star_positions:
                current_pair_dist = dist(self.expected_position, current_star_pos)

                if current_pair_dist < best_candidate_dist:
                    best_candidate_dist = current_pair_dist
                    new_star_pos = current_star_pos

            return new_star_pos

    def get_new_movement_vector(
        self,
        min_history_length: int = MIN_HISTORY_LENGTH,
        remove_outliers: bool = REMOVE_OUTLIERS,
        max_outlier_threshold: float = MAX_OUTLIER_THRESHOLD,
        default_vector: tuple[float, float] = DEFAULT_VECTOR,
        movement_vector_computation_method: str = MOVEMENT_VECTOR_COMPUTATION_METHOD,
    ) -> tuple[float, float]:
        """Function to calculate the star movement vector based on its last detected
        positions."""

        if len(self.last_positions) < min_history_length:
            return default_vector

        movement_vectors = self.get_individual_movement_vectors()
        mean_vector = get_average_vector(
            movement_vectors,
            default_vector,
            movement_vector_computation_method,
        )

        if not remove_outliers:
            return mean_vector

        filtered_vectors = [
            current_vector
            for current_vector in movement_vectors
            if dist(mean_vector, current_vector) < max_outlier_threshold
        ]

        return get_average_vector(
            filtered_vectors,
            default_vector,
            movement_vector_computation_method,
        )

    def get_individual_movement_vectors(self) -> list[tuple[float, float]]:
        """Function to get the individual movement vectors between each pair of its last
        detected positions."""

        for index, value in enumerate(self.last_positions):
            if value is not None:
                i_index = index
                break
        else:
            return []

        movement_vectors = []
        while True:
            j_index = i_index + 1

            while j_index < len(self.last_positions):
                if self.last_positions[j_index] is None:
                    j_index += 1
                else:
                    break
            if j_index >= len(self.last_positions):
                break

            number_of_frames_in_between = j_index - i_index
            current_movement_vector = (
                (self.last_positions[j_index][0] - self.last_positions[i_index][0])
                / number_of_frames_in_between,
                (self.last_positions[j_index][1] - self.last_positions[i_index][1])
                / number_of_frames_in_between,
            )
            movement_vectors.extend(
                [current_movement_vector] * number_of_frames_in_between
            )

            i_index = j_index

        return movement_vectors

    def get_new_expected_position(
        self,
        expected_position: tuple[int, int] = None,
    ) -> tuple[int, int]:
        """Function to calculate the expected position of the star."""

        return (
            (
                round(
                    self.last_detected_position[0]
                    + (self.movement_vector[0] * self.frames_since_last_detection)
                ),
                round(
                    self.last_detected_position[1]
                    + (self.movement_vector[1] * self.frames_since_last_detection)
                ),
            )
            if expected_position is None and self.last_detected_position is not None
            else expected_position
        )

    def get_new_last_detected_position(
        self,
        last_detected_position: tuple[int, int] = None,
    ) -> Optional[tuple[int, int]]:
        """
        Function to calculate the last detected position of the star and the number of
        frames since the last detection.
        <br/><br/>

        Note: This function modifies data from the 'frames_since_last_detection' class
        attribute for performance reduction purposes.
        """

        if last_detected_position is not None:
            return last_detected_position

        self.frames_since_last_detection = 0

        for pos in reversed(self.last_positions):
            self.frames_since_last_detection += 1

            if pos is not None:
                return pos

        return None


# ToDo: extract this to another file?
# ToDo: test the different methods and compare accuracy and performance
def get_average_vector(
    vectors: list[tuple[float, float]],
    default_vector: tuple[float, float] = DEFAULT_VECTOR,
    computation_method: str = MOVEMENT_VECTOR_COMPUTATION_METHOD,
) -> tuple[float, float]:
    """
    Function to calculate the average (or central tendency) vector from the given list.
    <br/><br/>

    The 'computation_method' parameter can be used to select how to calculate the
    resulting vector.
    <br/>
    - mean: arithmetic mean <br/>
    - median: middle value after sorting <br/>
    - mode: most common value <br/>
    <br/>

    If the given list of vectors is empty then the 'default_vector' is returned.
    """

    num_of_vectors = len(vectors)

    if num_of_vectors != 0:
        zipped_points = list(zip(*vectors))

        if computation_method == "mean":
            # return the mean of each axis
            return (
                sum(zipped_points[0]) / num_of_vectors,
                sum(zipped_points[1]) / num_of_vectors,
            )

        elif computation_method == "median":
            # return the median of each axis
            center_index = num_of_vectors // 2
            sorted_zipped_points = [
                sorted(zipped_points[0]),
                sorted(zipped_points[1]),
            ]

            if num_of_vectors % 2 == 0:
                return (
                    (
                        (
                            (sorted_zipped_points[0][center_index - 1])
                            + (sorted_zipped_points[0][center_index])
                        )
                        / 2
                    ),
                    (
                        (
                            (sorted_zipped_points[1][center_index - 1])
                            + (sorted_zipped_points[1][center_index])
                        )
                        / 2
                    ),
                )

            else:
                return (
                    sorted_zipped_points[0][center_index],
                    sorted_zipped_points[1][center_index],
                )

        elif computation_method == "mode":
            # return the mode of each axis
            return (
                max(set(zipped_points[0]), key=zipped_points[0].count),
                max(set(zipped_points[1]), key=zipped_points[1].count),
            )

        else:
            raise ValueError(
                "Error: Invalid computation method, "
                + "the available values are: 'mean', 'median' and 'mode'"
            )

    return default_vector
