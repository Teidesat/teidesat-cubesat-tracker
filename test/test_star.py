#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at src/image_processor.py
 file.
"""

import random
import unittest
from unittest.mock import Mock, MagicMock

import cv2 as cv

from src.star import (
    DEFAULT_LEFT_LIFETIME,
    VIDEO_FPS,
    DEFAULT_VECTOR,
    Star,
    id_generator,
    get_average_vector,
)


class StarClassTestCase(unittest.TestCase):
    """Class to test the image_processor script."""

    @classmethod
    def setUpClass(cls):
        cls.star_detector = cv.FastFeatureDetector_create(threshold=50)

    def test_id_generator(self):
        """The id generator gives a different integer number each time."""

        id_gen = id_generator()

        number_of_ids = 1000
        unique_ids = len({next(id_gen) for _ in range(number_of_ids)})
        expected_unique_ids = number_of_ids

        self.assertEqual(expected_unique_ids, unique_ids)

    def test_default_star(self):
        """A new default star can be created."""

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        expected_result = {
            "id": 0,
            "last_positions": [],
            "last_times_detected": [],
            "lifetime": 1,
            "left_lifetime": DEFAULT_LEFT_LIFETIME,
            "blinking_freq": VIDEO_FPS,
            "detection_confidence": 0,
            "movement_vector": DEFAULT_VECTOR,
            "color": [255, 0, 0],
            "frames_since_last_detection": 0,
            "last_detected_position": None,
            "next_expected_position": None,
            "last_predicted_position": None,
        }

        result = Star()
        self.assertEqual(expected_result, result.__dict__)

    def test_custom_star(self):
        """A new customized star can be created."""

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        expected_result = {
            "id": 0,
            "last_positions": [(10, 15)],
            "last_times_detected": [1],
            "lifetime": 3,
            "left_lifetime": 7,
            "blinking_freq": 15,
            "detection_confidence": -2,
            "movement_vector": (0, 0),
            "color": [0, 0, 100],
            "frames_since_last_detection": 1,
            "last_detected_position": (10, 15),
            "next_expected_position": (10, 15),
            "last_predicted_position": (10, 15),
        }

        result = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
            color=[0, 0, 100],
        )

        self.assertEqual(expected_result, result.__dict__)

    def test_eq_1(self):
        """__eq__ can detect two identical stars."""

        star_1 = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
            color=[0, 0, 100],
        )

        star_2 = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
            color=[0, 0, 100],
        )

        self.assertEqual(star_1, star_2)

    def test_eq_2(self):
        """__eq__ can detect two different stars."""

        star_1 = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
            color=[0, 0, 100],
        )

        star_2 = Star(
            last_positions=[(11, 16), None, None],
            last_times_detected=[1, 0, 0],
            lifetime=3,
            left_lifetime=8,
            detection_confidence=-6,
        )

        self.assertNotEqual(star_1, star_2)

    def test_repr(self):
        """__repr__ can return the string representation of a star."""

        star = Star()
        expected_result = (
            f"<{star.__class__.__module__}.{star.__class__.__name__}"
            + f" object at {hex(id(star))}>: "
            + str(
                {
                    "id": 0,
                    "last_positions": [],
                    "last_times_detected": [],
                    "lifetime": 1,
                    "left_lifetime": DEFAULT_LEFT_LIFETIME,
                    "blinking_freq": VIDEO_FPS,
                    "detection_confidence": 0,
                    "movement_vector": DEFAULT_VECTOR,
                    "color": [255, 0.0, 0.0],
                    "frames_since_last_detection": 0,
                    "last_detected_position": None,
                    "next_expected_position": None,
                    "last_predicted_position": None,
                }
            )
        )
        self.assertEqual(expected_result, star.__repr__())

    def test_update_info_1(self):
        """update_info can add the new star position and update the star information
        accordingly."""

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        star_positions = [(11, 16)]
        star = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
        )

        default_left_lifetime = 20
        video_fps = 30
        detected_stars = {star}
        expected_positions = []
        expected_stars = {
            Star(
                last_positions=[(10, 15), (11, 16)],
                last_times_detected=[1, 1],
                lifetime=4,
                left_lifetime=default_left_lifetime,
                blinking_freq=video_fps,
                detection_confidence=-1,
                movement_vector=(1, 1),
                last_predicted_position=(10, 15),
            )
        }

        star.update_info(
            star_positions,
            detected_stars,
            sat_desired_blinking_freq=15,
            video_fps=video_fps,
            default_left_lifetime=default_left_lifetime,
            max_history_length=5,
            min_history_length=1,
            frequency_threshold=20,
        )

        self.assertEqual(expected_positions, star_positions)
        self.assertEqual(expected_stars, detected_stars)

    def test_update_info_2(self):
        """update_info can reduce the lifetime of the star if it has no new position and
        update its information accordingly."""

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        star_positions = []
        star = Star(
            last_positions=[(10, 15), None, None],
            last_times_detected=[1, 0, 0],
            lifetime=3,
            left_lifetime=8,
            detection_confidence=-6,
        )

        detected_stars = {star}
        expected_positions = []
        expected_stars = {
            Star(
                last_positions=[None, None, None],
                last_times_detected=[0, 0, 0],
                lifetime=4,
                left_lifetime=7,
                blinking_freq=0,
                detection_confidence=-8,
                frames_since_last_detection=4,
                last_detected_position=(10, 15),
            )
        }

        star.update_info(star_positions, detected_stars, max_history_length=3)

        self.assertEqual(expected_positions, star_positions)
        self.assertEqual(expected_stars, detected_stars)

    def test_update_info_3(self):
        """update_info can forget a star that has no left lifetime."""

        star = Star(left_lifetime=0)
        star_positions = []
        detected_stars = {star}
        expected_positions = []
        expected_stars = set()

        star.update_info(star_positions, detected_stars)

        self.assertEqual(expected_positions, star_positions)
        self.assertEqual(expected_stars, detected_stars)

    def test_get_new_star_position_1(self):
        """get_new_star_position can get the new position if it coincides with the
        expected one."""

        star_positions = [
            (5, 5),
            (7, 12),
            (10, 15),
        ]
        star = Star(
            last_positions=[(7, 12)],
            movement_vector=(0, 0),
        )
        expected_result = (7, 12)

        result = star.get_new_star_position(star_positions)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_2(self):
        """get_new_star_position can get the closest new position to the expected
        one."""

        star_positions = [
            (5, 5),
            (8, 14),
            (10, 15),
        ]
        star = Star(
            last_positions=[(7, 12)],
            movement_vector=(1, 1),
        )
        expected_result = (8, 14)

        result = star.get_new_star_position(star_positions, max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_3(self):
        """get_new_star_position can't get any new position if there are no stars in
        range."""

        star_positions = [
            (5, 5),
            (9, 20),
            (10, 15),
        ]
        star = Star(
            last_positions=[(12, 17)],
            movement_vector=(2, 3),
        )
        expected_result = None

        result = star.get_new_star_position(star_positions, max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_4(self):
        """get_new_star_position can't get any new position if there are no new
        stars."""

        star_positions = []
        star = Star(
            last_positions=[(12, 17)],
            movement_vector=(2, 3),
        )
        expected_result = None

        result = star.get_new_star_position(star_positions, max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_movement_vector_1(self):
        """get_new_movement_vector can return a default movement vector if not enough
        points are given."""

        star = Star(last_positions=[])
        expected_result = (0, 0)

        result = star.get_new_movement_vector(default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_new_movement_vector_2(self):
        """get_new_movement_vector can get the movement vector from a list of
        positions."""

        star = Star(
            last_positions=[
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
            ]
        )
        expected_result = (1, 0)

        result = star.get_new_movement_vector(
            min_history_length=1,
            remove_outliers=False,
            movement_vector_computation_method="mean",
        )
        self.assertEqual(expected_result, result)

    def test_get_new_movement_vector_3(self):
        """get_new_movement_vector can remove an outlier point and get the movement
        vector of the remaining positions."""

        star = Star(
            last_positions=[
                (1, 4),
                (2, 4),
                (3, 4),
                (53, 487),
                (5, 4),
            ]
        )
        expected_result = (1, 0)

        result = star.get_new_movement_vector(
            min_history_length=1,
            max_outlier_threshold=1.5,
            remove_outliers=True,
            movement_vector_computation_method="mean",
        )
        self.assertEqual(expected_result, result)

    def test_get_new_movement_vector_4(self):
        """get_new_movement_vector can return a default movement vector if all positions
        are treated as outliers."""

        star = Star(
            last_positions=[
                (1, 4),
                (4, 6),
                (7, 4),
                (5, 9),
                (2, 1),
            ]
        )
        expected_result = (0, 0)

        result = star.get_new_movement_vector(
            min_history_length=1,
            max_outlier_threshold=0.1,
            remove_outliers=True,
            default_vector=(0, 0),
            movement_vector_computation_method="mean",
        )
        self.assertEqual(expected_result, result)

    def test_get_new_movement_vector_5(self):
        """get_new_movement_vector can get the movement vector even with lost
        positions."""

        star = Star(
            last_positions=[
                None,
                (1, 4),
                None,
                (3, 4),
                (4, 4),
                None,
                None,
                (7, 4),
                None,
            ]
        )
        expected_result = (1, 0)

        result = star.get_new_movement_vector(
            min_history_length=1,
            remove_outliers=False,
            movement_vector_computation_method="mean",
        )
        self.assertEqual(expected_result, result)

    def test_get_individual_movement_vectors_1(self):
        """get_individual_movement_vectors can get the individual movement vectors from
        two detected positions."""

        star = Star(
            last_positions=[
                (1, 4),
                (2, 4),
            ]
        )
        expected_result = [(1, 0)]

        result = star.get_individual_movement_vectors()
        self.assertEqual(expected_result, result)

    def test_get_individual_movement_vectors_2(self):
        """get_individual_movement_vectors can get the individual movement vectors with
        one undetected position in between."""

        star = Star(
            last_positions=[
                (1, 4),
                None,
                (3, 4),
            ]
        )
        expected_result = [(1, 0), (1, 0)]

        result = star.get_individual_movement_vectors()
        self.assertEqual(expected_result, result)

    def test_get_individual_movement_vectors_3(self):
        """get_individual_movement_vectors can return an empty list if there are not
        enough detected positions."""

        star = Star(
            last_positions=[
                (2, 4),
            ]
        )
        expected_result = []

        result = star.get_individual_movement_vectors()
        self.assertEqual(expected_result, result)

    def test_get_individual_movement_vectors_4(self):
        """get_individual_movement_vectors can return an empty list if there are not
        enough detected positions."""

        star = Star(
            last_positions=[
                None,
            ]
        )
        expected_result = []

        result = star.get_individual_movement_vectors()
        self.assertEqual(expected_result, result)

    def test_get_individual_movement_vectors_5(self):
        """get_individual_movement_vectors can return an empty list if there are not
        enough detected positions."""

        star = Star(
            last_positions=[
                None,
                (2, 4),
                None,
                None,
            ]
        )
        expected_result = []

        result = star.get_individual_movement_vectors()
        self.assertEqual(expected_result, result)

    def test_next_expected_position_1(self):
        """next_expected_position can return the expected position of a static star."""

        star = Star(
            movement_vector=(0, 0),
            last_detected_position=(10, 15),
            frames_since_last_detection=1,
        )
        expected_result = (10, 15)

        self.assertEqual(expected_result, star.next_expected_position)

    def test_next_expected_position_2(self):
        """next_expected_position can return the expected position of a moving star."""

        star = Star(
            movement_vector=(1, 1),
            last_detected_position=(10, 15),
            frames_since_last_detection=1,
        )
        expected_result = (11, 16)

        self.assertEqual(expected_result, star.next_expected_position)

    def test_next_expected_position_3(self):
        """next_expected_position can return the expected position of a lost moving star."""

        star = Star(
            movement_vector=(1, 1),
            last_detected_position=(10, 15),
            frames_since_last_detection=3,
        )
        expected_result = (13, 18)

        self.assertEqual(expected_result, star.next_expected_position)

    def test_last_predicted_position_1(self):
        """last_predicted_position can return the last predicted position of a static star."""

        star = Star(
            last_positions=[(10, 15)],
            movement_vector=(0, 0),
        )
        expected_result = (10, 15)

        self.assertEqual(expected_result, star.last_predicted_position)

    def test_last_predicted_position_2(self):
        """last_predicted_position can return the last predicted position of a moving star."""

        star = Star(
            last_positions=[(10, 15)],
            movement_vector=(1, 1),
        )
        expected_result = (11, 16)

        self.assertEqual(expected_result, star.last_predicted_position)

    def test_last_detected_position_1(self):
        """last_detected_position can return the last detected position of the star."""

        star = Star(
            last_positions=[(10, 15)],
        )
        expected_result = (10, 15)

        self.assertEqual(expected_result, star.last_detected_position)

    def test_last_detected_position_2(self):
        """last_detected_position can return the last detected position of the star
        ignoring the not detected ones."""

        star = Star(
            last_positions=[(10, 15), None, None],
        )
        expected_result = (10, 15)

        self.assertEqual(expected_result, star.last_detected_position)

    def test_last_detected_position_3(self):
        """last_detected_position can return None if the star has no position."""

        star = Star(
            last_positions=[],
        )
        expected_result = None

        self.assertEqual(expected_result, star.last_detected_position)

    def test_last_detected_position_4(self):
        """last_detected_position can return None if the star has no detected
        positions."""

        star = Star(
            last_positions=[None, None, None],
        )
        expected_result = None

        self.assertEqual(expected_result, star.last_detected_position)

    def test_get_average_vect_1(self):
        """get_average_vect can return the default vector if an empty list is given."""

        vectors = []
        expected_result = (0, 0)

        result = get_average_vector(vectors, default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_average_vect_2(self):
        """get_average_vect can get the mean vector from a list of vectors."""

        vectors = [
            (3.2, 7.1),
            (2.1, 5.9),
            (1.8, 6.2),
            (2.0, 5.8),
            (1.8, 6.5),
            (2.3, 6.8),
            (2.5, 5.9),
        ]
        expected_result = (2.24, 6.31)

        result = get_average_vector(vectors, computation_method="mean")
        self.assertAlmostEqual(expected_result[0], result[0], places=2)

    def test_get_average_vect_3(self):
        """get_average_vect can get the median vector from a list of vectors containing
        an odd number of items."""

        vectors = [
            (3.2, 7.1),
            (2.1, 5.9),
            (1.8, 6.2),
            (2.0, 5.8),
            (1.8, 6.5),
            (2.3, 6.8),
            (2.5, 5.9),
        ]
        expected_result = (2.1, 6.2)

        result = get_average_vector(vectors, computation_method="median")
        self.assertEqual(expected_result, result)

    def test_get_average_vect_4(self):
        """get_average_vect can get the median vector from a list of vectors containing
        an even number of items."""

        vectors = [
            (3.2, 7.1),
            (2.1, 5.9),
            (1.8, 6.2),
            (2.0, 5.8),
            (1.8, 6.5),
            (2.3, 6.8),
            (2.5, 5.9),
            (2.9, 6.3),
        ]
        expected_result = (2.2, 6.25)

        result = get_average_vector(vectors, computation_method="median")
        self.assertEqual(expected_result, result)

    def test_get_average_vect_5(self):
        """get_average_vect can get the mode vector from a list of vectors."""

        vectors = [
            (3.2, 7.1),
            (2.1, 5.9),
            (1.8, 6.2),
            (2.0, 5.8),
            (1.8, 6.5),
            (2.3, 6.8),
            (2.5, 5.9),
            (2.9, 6.3),
        ]
        expected_result = (1.8, 5.9)

        result = get_average_vector(vectors, computation_method="mode")
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
