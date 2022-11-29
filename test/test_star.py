#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at
 src/image_processor.py file.
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
    get_mean_vector,
)


class DataGettersTestCase(unittest.TestCase):
    """ Class to test the image_processor script. """

    @classmethod
    def setUpClass(cls):
        cls.star_detector = cv.FastFeatureDetector_create(threshold=50)

    def test_default_star(self):
        """ A new default star can be created. """

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
        }

        result = Star()
        self.assertEqual(expected_result, result.__dict__)

    def test_custom_star(self):
        """ A new customized star can be created. """

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

    def test_update_info_1(self):
        """ update_info can add the new star position and update the star
        information accordingly. """

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        star_positions = [(11, 16)]
        star = Star(
            last_positions=[(10, 15)],
            last_times_detected=[0],
            lifetime=3,
            left_lifetime=7,
            blinking_freq=15,
            detection_confidence=-2,
            movement_vector=(0, 0),
        )

        default_left_lifetime = 20
        video_fps = 30
        detected_stars = {0: star}
        expected_star = Star(
            last_positions=[(10, 15), (11, 16)],
            last_times_detected=[0, 1],
            lifetime=4,
            left_lifetime=default_left_lifetime,
            blinking_freq=(video_fps / 2),
            detection_confidence=-1,
            movement_vector=(1, 1)
        )
        expected_positions = []
        expected_stars = {0: expected_star}

        star.update_info(star_positions,
                         detected_stars,
                         sat_desired_blinking_freq=15,
                         video_fps=video_fps,
                         default_left_lifetime=default_left_lifetime,
                         max_history_length=5,
                         min_history_length=1,
                         frequency_threshold=20)

        self.assertEqual(expected_positions, star_positions)
        self.assertEqual(expected_stars.keys(), detected_stars.keys())

        for expected_star, detected_star in zip(expected_stars.values(),
                                                detected_stars.values()):
            self.assertEqual(expected_star.__dict__, detected_star.__dict__)

    def test_update_info_2(self):
        """ update_info can reduce the lifetime of the star if it has no new
        position and update its information accordingly. """

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        star_positions = []
        star = Star(
            last_positions=[(10, 15)],
            last_times_detected=[1, 0, 0],
            lifetime=3,
            left_lifetime=8,
            detection_confidence=-6,
        )

        detected_stars = {0: star}
        expected_star = Star(
            last_positions=[(10, 15)],
            last_times_detected=[0, 0, 0],
            lifetime=4,
            left_lifetime=7,
            blinking_freq=0,
            detection_confidence=-8,
        )
        expected_result = {0: expected_star}

        star.update_info(star_positions, detected_stars, max_history_length=3)
        self.assertEqual(expected_result.keys(), detected_stars.keys())

        for expected_star, detected_star in zip(expected_result.values(),
                                                detected_stars.values()):
            self.assertEqual(expected_star.__dict__, detected_star.__dict__)

    def test_update_info_3(self):
        """ update_info can forget a star that has no left lifetime. """

        star = Star(left_lifetime=0)
        star_positions = []
        detected_stars = {0: star}
        expected_result = {}

        star.update_info(star_positions, detected_stars)
        self.assertEqual(expected_result, detected_stars)

    def test_get_new_star_position_1(self):
        """ get_new_star_position can get the new position if it coincides with
        the expected one. """

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
        """ get_new_star_position can get the closest new position to the
        expected one. """

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

        result = star.get_new_star_position(star_positions,
                                            max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_3(self):
        """ get_new_star_position can't get any new position if there are no
        stars in range. """

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

        result = star.get_new_star_position(star_positions,
                                            max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_4(self):
        """ get_new_star_position can't get any new position if there are no
        new stars. """

        star_positions = []
        star = Star(
            last_positions=[(12, 17)],
            movement_vector=(2, 3),
        )
        expected_result = None

        result = star.get_new_star_position(star_positions,
                                            max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_1(self):
        """ get_movement_vector can return a default movement vector if not
        enough points are given. """

        star = Star(last_positions=[])
        expected_result = (0, 0)

        result = star.get_movement_vector(default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_2(self):
        """ get_movement_vector can get the movement vector from a list of
        positions. """

        star = Star(last_positions=[
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
        ])
        expected_result = (1, 0)

        result = star.get_movement_vector(min_history_length=1,
                                          remove_outliers=False)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_3(self):
        """ get_movement_vector can remove an outlier point and get the
        movement vector of the remaining positions. """

        star = Star(last_positions=[
            (1, 4),
            (2, 4),
            (3, 4),
            (53, 487),
            (5, 4),
        ])
        expected_result = (1, 0)

        result = star.get_movement_vector(min_history_length=1,
                                          max_outlier_threshold=1.5,
                                          remove_outliers=True)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_4(self):
        """ get_movement_vector can return a default movement vector if all
        positions are treated as outliers. """

        star = Star(last_positions=[
            (1, 4),
            (4, 6),
            (7, 4),
            (5, 9),
            (2, 1),
        ])
        expected_result = (0, 0)

        result = star.get_movement_vector(min_history_length=1,
                                          remove_outliers=True,
                                          max_outlier_threshold=0.1,
                                          default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_mean_vect_1(self):
        """ get_mean_vect can return a default vector if not enough points are
        given. """

        points = []
        expected_result = (0, 0)

        result = get_mean_vector(points, default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_mean_vect_2(self):
        """ get_mean_vect can get the mean vector from a list of points. """

        points = [
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
        ]
        expected_result = (3, 4)

        result = get_mean_vector(points, default_vector=(0, 0))
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
