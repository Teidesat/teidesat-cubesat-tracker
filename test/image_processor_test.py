#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at
 src/image_processor/image_processor.py file.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import random

import cv2 as cv

from src.image_processor.image_processor import (
    DEFAULT_LEFT_LIFETIME,
    detect_stars,
    prune_close_points,
    track_stars,
    add_remaining_stars,
    get_new_star_position,
    get_movement_vector,
    get_mean_vect,
    detect_shooting_stars,
    detect_blinking_star,
)


class DataGettersTestCase(unittest.TestCase):
    """ Class to test the image_processor script. """

    @classmethod
    def setUpClass(cls):
        cls.star_detector = cv.FastFeatureDetector_create(threshold=50)

    def test_detect_stars_1(self):
        """ detect_stars can detect one star. """

        image = cv.imread(str(Path("./data/images/stellarium-007.png")))

        if image is None:
            sys.exit("Could not read the image.")

        expected_result = [(17, 17)]

        result = detect_stars(image, self.star_detector)
        self.assertEqual(expected_result, result)

    def test_detect_stars_2(self):
        """ detect_stars can detect multiple stars. """

        image = cv.imread(str(Path("./data/images/stellarium-003.png")))

        if image is None:
            sys.exit("Could not read the image.")

        expected_result = [
            (386, 11),
            (420, 100),
            (248, 120),
            (160, 138),
            (90, 147),
            (304, 171),
            (16, 223),
        ]

        result = detect_stars(image, self.star_detector)
        self.assertEqual(expected_result, result)

    def test_prune_close_points_1(self):
        """ prune_close_points can filter a duplicate star. """

        star_positions = [
            (8, 8),
            (10, 10),
        ]
        expected_result = [(10, 10)]

        result = prune_close_points(star_positions, min_prune_distance=5)
        self.assertEqual(expected_result, result)

    def test_prune_close_points_2(self):
        """ prune_close_points can ignore a distant star . """

        star_positions = [
            (8, 8),
            (10, 10),
            (15, 15),
        ]
        expected_result = [
            (10, 10),
            (15, 15)
        ]

        result = prune_close_points(star_positions, min_prune_distance=5)
        self.assertEqual(expected_result, result)

    def test_track_stars_0(self):
        """ track_stars can do nothing if no stars are given. """

        star_positions = []
        detected_stars = {}
        expected_result = {}

        track_stars(star_positions, detected_stars)
        self.assertEqual(expected_result, detected_stars)

    def test_track_stars_1(self):
        """ track_stars can add a new star. """

        random.random = Mock(return_value=1)

        video_fps = 60
        desired_blinking_freq = 30
        star_positions = [(10, 15)]
        detected_stars = {}
        expected_result = {
            0: {
                "last_positions": [(10, 15)],
                "last_times_detected": [1],
                "lifetime": 1,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": video_fps,
                "detection_confidence": 0,
                "movement_vector": (0, 0),
                "color": [255, 0, 0],
            }
        }

        track_stars(star_positions,
                    detected_stars,
                    desired_blinking_freq,
                    video_fps)

        self.assertEqual(expected_result, detected_stars)

    def test_track_stars_2(self):
        """ track_stars can keep track of a moving star. """

        video_fps = 60
        desired_blinking_freq = 30
        star_positions = [(11, 16)]
        detected_stars = {
            0: {
                "last_positions": [(10, 15)],
                "last_times_detected": [1, 0, 0],
                "lifetime": 3,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": video_fps / 3,
                "detection_confidence": 0,
                "movement_vector": (0, 0),
                "color": [],
            }
        }
        expected_result = {
            0: {
                "last_positions": [(10, 15), (11, 16)],
                "last_times_detected": [1, 0, 0, 1],
                "lifetime": 4,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": video_fps / 2,
                "detection_confidence": 1,
                "movement_vector": (0, 0),
                "color": [],
            }
        }

        track_stars(star_positions,
                    detected_stars,
                    desired_blinking_freq,
                    video_fps)

        self.assertEqual(expected_result, detected_stars)

    def test_track_stars_3(self):
        """ track_stars can remember a hidden star. """

        video_fps = 60
        desired_blinking_freq = 30
        star_positions = []
        detected_stars = {
            0: {
                "last_positions": [(11, 16)],
                "last_times_detected": [1, 1],
                "lifetime": 2,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": video_fps,
                "detection_confidence": 0,
                "movement_vector": (0, 0),
                "color": [],
            }
        }
        expected_result = {
            0: {
                "last_positions": [(11, 16)],
                "last_times_detected": [1, 1, 0],
                "lifetime": 3,
                "left_lifetime": DEFAULT_LEFT_LIFETIME - 1,
                "blinking_freq": (2 / 3) * video_fps,
                "detection_confidence": -2,
                "movement_vector": (0, 0),
                "color": [],
            }
        }

        track_stars(star_positions,
                    detected_stars,
                    desired_blinking_freq,
                    video_fps)

        self.assertEqual(expected_result, detected_stars)

    def test_track_stars_4(self):
        """ track_stars can forget a star not found for a long time. """

        star_positions = []
        detected_stars = {
            0: {
                "last_positions": [(11, 16)],
                "left_lifetime": 0,
                "movement_vector": (0, 0),
            }
        }
        expected_result = {}

        track_stars(star_positions, detected_stars)
        self.assertEqual(expected_result, detected_stars)

    def test_add_remaining_stars_0(self):
        """ add_remaining_stars can do nothing if no stars are given. """

        star_positions = []
        detected_stars = {}
        expected_result = {}

        add_remaining_stars(star_positions, detected_stars)
        self.assertEqual(expected_result, detected_stars)

    @patch('src.image_processor.image_processor.new_star_id', 0)
    def test_add_remaining_stars_1(self):
        """ add_remaining_stars can add one star to the detected stars'
        dictionary. """

        random.random = Mock(return_value=1)

        video_fps = 60
        default_left_lifetime = 10
        default_movement_vector = (0, 0)
        star_positions = [(11, 16)]
        detected_stars = {}
        expected_result = {
            0: {
                "last_positions": [(11, 16)],
                "last_times_detected": [1],
                "lifetime": 1,
                "left_lifetime": default_left_lifetime,
                "blinking_freq": video_fps,
                "detection_confidence": 0,
                "movement_vector": default_movement_vector,
                "color": [255, 0, 0],
            }
        }

        add_remaining_stars(star_positions,
                            detected_stars,
                            video_fps,
                            default_left_lifetime,
                            default_movement_vector)

        self.assertEqual(expected_result, detected_stars)

    @patch('src.image_processor.image_processor.new_star_id', 0)
    def test_add_remaining_stars_2(self):
        """ add_remaining_stars can add multiple stars to the detected stars'
        dictionary. """

        random.random = Mock(return_value=1)

        video_fps = 60
        default_left_lifetime = 10
        default_movement_vector = (0, 0)
        star_positions = [(11, 16), (7, 12)]
        detected_stars = {}
        expected_result = {
            0: {
                "last_positions": [(11, 16)],
                "last_times_detected": [1],
                "lifetime": 1,
                "left_lifetime": default_left_lifetime,
                "blinking_freq": video_fps,
                "detection_confidence": 0,
                "movement_vector": default_movement_vector,
                "color": [255, 0, 0],
            },
            1: {
                "last_positions": [(7, 12)],
                "last_times_detected": [1],
                "lifetime": 1,
                "left_lifetime": default_left_lifetime,
                "blinking_freq": video_fps,
                "detection_confidence": 0,
                "movement_vector": default_movement_vector,
                "color": [255, 0, 0],
            }
        }

        add_remaining_stars(star_positions,
                            detected_stars,
                            video_fps,
                            default_left_lifetime,
                            default_movement_vector)

        self.assertEqual(expected_result, detected_stars)

    def test_get_new_star_position_1(self):
        """ get_new_star_position can get the new position if it coincides with
        the expected one. """

        star_positions = [
            (5, 5),
            (7, 12),
            (10, 15),
        ]
        old_star_info = {
            "last_positions": [(7, 12)],
            "movement_vector": (0, 0),
        }
        expected_result = (7, 12)

        result = get_new_star_position(star_positions,
                                       old_star_info,
                                       max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_2(self):
        """ get_new_star_position can get the closest new position to the
        expected one. """

        star_positions = [
            (5, 5),
            (8, 14),
            (10, 15),
        ]
        old_star_info = {
            "last_positions": [(7, 12)],
            "movement_vector": (1, 1),
        }
        expected_result = (8, 14)

        result = get_new_star_position(star_positions,
                                       old_star_info,
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
        old_star_info = {
            "last_positions": [(12, 17)],
            "movement_vector": (2, 3),
        }
        expected_result = None

        result = get_new_star_position(star_positions,
                                       old_star_info,
                                       max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_new_star_position_4(self):
        """ get_new_star_position can't get any new position if there are no
        new stars. """

        star_positions = []
        old_star_info = {
            "last_positions": [(12, 17)],
            "movement_vector": (2, 3),
        }
        expected_result = None

        result = get_new_star_position(star_positions,
                                       old_star_info,
                                       max_move_distance=5)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_1(self):
        """ get_movement_vector can return a default movement vector if not
        enough points are given. """

        last_positions = []
        expected_result = (0, 0)

        result = get_movement_vector(last_positions,
                                     default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_2(self):
        """ get_movement_vector can get the movement vector from a list of
        positions. """

        last_positions = [
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
        ]
        expected_result = (1, 0)

        result = get_movement_vector(last_positions,
                                     min_history_length=1,
                                     remove_outliers=False)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_3(self):
        """ get_movement_vector can remove an outlier point and get the
        movement vector of the remaining positions. """

        last_positions = [
            (1, 4),
            (2, 4),
            (3, 4),
            (53, 487),
            (5, 4),
        ]
        expected_result = (1, 0)

        result = get_movement_vector(last_positions,
                                     min_history_length=1,
                                     remove_outliers=True)
        self.assertEqual(expected_result, result)

    def test_get_movement_vector_4(self):
        """ get_movement_vector can return a default movement vector if all
        positions are treated as outliers. """

        last_positions = [
            (1, 4),
            (4, 6),
            (7, 4),
            (5, 9),
            (2, 1),
        ]
        expected_result = (0, 0)

        result = get_movement_vector(last_positions,
                                     min_history_length=1,
                                     remove_outliers=True,
                                     max_outlier_threshold=0.1,
                                     default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_get_mean_vect_1(self):
        """ get_mean_vect can return a default vector if not enough points are
        given. """

        points = []
        expected_result = (0, 0)

        result = get_mean_vect(points, default_vector=(0, 0))
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

        result = get_mean_vect(points, default_vector=(0, 0))
        self.assertEqual(expected_result, result)

    def test_detect_shooting_stars(self):
        """ detect_shooting_stars can filter the detected stars to get the
        shooting stars or satellites only. """

        detected_stars = {
            0: {"movement_vector": (1, 0)},
            1: {"movement_vector": (0.5, -1.3)},
            2: {"movement_vector": (2.1, 0.7)},
            3: {"movement_vector": (-1.4, 3.3)},
            4: {"movement_vector": (0.4, -0.1)},
        }
        expected_result = {
            2: {"movement_vector": (2.1, 0.7)},
            3: {"movement_vector": (-1.4, 3.3)},
        }

        result = detect_shooting_stars(detected_stars, movement_threshold=2)
        self.assertEqual(expected_result, result)

    def test_detect_blinking_star_1(self):
        """ detect_blinking_star can detect the star that has the highest
        confidence of being the satellite. """

        detected_stars = {
            0: {"detection_confidence": 6},
            1: {"detection_confidence": 485},
            2: {"detection_confidence": 8342},
            3: {"detection_confidence": 141},
            4: {"detection_confidence": 48},
        }
        expected_result = (2, {"detection_confidence": 8342})

        result = detect_blinking_star(detected_stars,
                                      min_detection_confidence=20)
        self.assertEqual(expected_result, result)

    def test_detect_blinking_star_2(self):
        """ detect_blinking_star can ignore all stars if they have not enough
        confidence. """

        detected_stars = {
            0: {"detection_confidence": 6},
            1: {"detection_confidence": 15},
            2: {"detection_confidence": 3},
            3: {"detection_confidence": 7},
            4: {"detection_confidence": 18},
        }
        expected_result = None

        result = detect_blinking_star(detected_stars,
                                      min_detection_confidence=20)
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
