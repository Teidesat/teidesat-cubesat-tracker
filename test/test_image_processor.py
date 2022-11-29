#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at
 src/image_processor.py file.
"""

from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock, MagicMock

import cv2 as cv

from src.image_processor import (
    detect_stars,
    prune_close_points,
    track_stars,
    detect_shooting_stars,
    detect_blinking_star,
)
from src.star import Star


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
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        sat_desired_blinking_freq = 30
        video_fps = 60
        default_left_lifetime = 20
        default_movement_vector = (0, 0)

        star_positions = [(10, 15)]
        detected_stars = {}
        expected_result = {
            0: Star(
                last_positions=[(10, 15)],
                last_times_detected=[1],
                lifetime=1,
                left_lifetime=default_left_lifetime,
                blinking_freq=video_fps,
                detection_confidence=0,
                movement_vector=default_movement_vector,
                color=[255, 0, 0],
            )
        }

        track_stars(star_positions,
                    detected_stars,
                    sat_desired_blinking_freq=sat_desired_blinking_freq,
                    video_fps=video_fps,
                    default_left_lifetime=default_left_lifetime,
                    default_vector=default_movement_vector)

        self.assertEqual(expected_result.keys(), detected_stars.keys())

        for expected_star, detected_star in zip(expected_result.values(),
                                                detected_stars.values()):
            self.assertEqual(expected_star.__dict__, detected_star.__dict__)

    def test_track_stars_2(self):
        """ track_stars can keep track of a moving star. """

        sat_desired_blinking_freq = 30
        video_fps = 60
        default_left_lifetime = 20
        star_positions = [(11, 16)]
        detected_stars = {
            0: Star(
                last_positions=[(10, 15)],
                last_times_detected=[1, 0, 0],
                lifetime=3,
                left_lifetime=7,
                blinking_freq=video_fps / 3,
                detection_confidence=0,
                movement_vector=(0, 0),
                color=[],
            )
        }
        expected_result = {
            0: Star(
                last_positions=[(10, 15), (11, 16)],
                last_times_detected=[1, 0, 0, 1],
                lifetime=4,
                left_lifetime=default_left_lifetime,
                blinking_freq=video_fps / 2,
                detection_confidence=1,
                movement_vector=(0, 0),
                color=[],
            )
        }

        track_stars(star_positions,
                    detected_stars,
                    sat_desired_blinking_freq=sat_desired_blinking_freq,
                    video_fps=video_fps,
                    default_left_lifetime=default_left_lifetime)

        self.assertEqual(expected_result.keys(), detected_stars.keys())

        for expected_star, detected_star in zip(expected_result.values(),
                                                detected_stars.values()):
            self.assertEqual(expected_star.__dict__, detected_star.__dict__)

    def test_track_stars_3(self):
        """ track_stars can remember a hidden star. """

        sat_desired_blinking_freq = 30
        video_fps = 60
        star_positions = []
        detected_stars = {
            0: Star(
                last_positions=[(11, 16)],
                last_times_detected=[1, 1],
                lifetime=2,
                left_lifetime=8,
                blinking_freq=video_fps,
                detection_confidence=0,
                movement_vector=(0, 0),
                color=[],
            )
        }
        expected_result = {
            0: Star(
                last_positions=[(11, 16)],
                last_times_detected=[1, 1, 0],
                lifetime=3,
                left_lifetime=7,
                blinking_freq=(2 / 3) * video_fps,
                detection_confidence=-2,
                movement_vector=(0, 0),
                color=[],
            )
        }

        track_stars(star_positions,
                    detected_stars,
                    sat_desired_blinking_freq=sat_desired_blinking_freq,
                    video_fps=video_fps)

        self.assertEqual(expected_result.keys(), detected_stars.keys())

        for expected_star, detected_star in zip(expected_result.values(),
                                                detected_stars.values()):
            self.assertEqual(expected_star.__dict__, detected_star.__dict__)

    def test_track_stars_4(self):
        """ track_stars can forget a star not found for a long time. """

        star_positions = []
        detected_stars = {
            0: Star(
                last_positions=[(11, 16)],
                left_lifetime=0,
                movement_vector=(0, 0),
            )
        }
        expected_result = {}

        track_stars(star_positions, detected_stars)
        self.assertEqual(expected_result, detected_stars)

    def test_detect_shooting_stars_1(self):
        """ detect_shooting_stars can filter the detected stars to get the
        shooting stars or satellites only. """

        random.random = Mock(return_value=1)
        Star._id = MagicMock()
        Star._id.__next__.return_value = 0

        detected_stars = {
            0: Star(movement_vector=(1, 0)),
            1: Star(movement_vector=(0.5, -1.3)),
            2: Star(movement_vector=(2.1, 0.7)),
            3: Star(movement_vector=(-1.4, 3.3)),
            4: Star(movement_vector=(0.4, -0.1)),
        }
        expected_result = {
            2: Star(movement_vector=(2.1, 0.7)),
            3: Star(movement_vector=(-1.4, 3.3)),
        }

        result = detect_shooting_stars(detected_stars, movement_threshold=2)

        self.assertEqual(expected_result.keys(), result.keys())

        for expected_star, result_star in zip(expected_result.values(),
                                              result.values()):
            self.assertEqual(expected_star.__dict__, result_star.__dict__)

    def test_detect_shooting_stars_2(self):
        """ detect_shooting_stars can ignore all stars if they have a not big
        enough movement vector. """

        detected_stars = {
            0: Star(movement_vector=(1, 0)),
            1: Star(movement_vector=(0.5, -1.3)),
            2: Star(movement_vector=(0.1, 0.7)),
            3: Star(movement_vector=(-0.4, 1.3)),
            4: Star(movement_vector=(0.4, -0.1)),
        }
        expected_result = {}

        result = detect_shooting_stars(detected_stars, movement_threshold=2)

        self.assertEqual(expected_result, result)

    def test_detect_blinking_star_1(self):
        """ detect_blinking_star can detect the star that has the highest
        confidence of being the satellite. """

        detected_stars = {
            0: Star(detection_confidence=6),
            1: Star(detection_confidence=485),
            2: Star(detection_confidence=8342),
            3: Star(detection_confidence=141),
            4: Star(detection_confidence=48),
        }

        star_id, blinking_star = detect_blinking_star(
            detected_stars,
            min_detection_confidence=20
        )

        self.assertEqual(2, star_id)
        self.assertEqual(8342, blinking_star.detection_confidence)

    def test_detect_blinking_star_2(self):
        """ detect_blinking_star can ignore all stars if they have not enough
        confidence. """

        detected_stars = {
            0: Star(detection_confidence=6),
            1: Star(detection_confidence=15),
            2: Star(detection_confidence=3),
            3: Star(detection_confidence=7),
            4: Star(detection_confidence=18),
        }
        expected_result = None

        result = detect_blinking_star(detected_stars,
                                      min_detection_confidence=20)
        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
