#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at
 src/image_processor/image_processor.py file.
"""

from pathlib import Path
import sys
import unittest

import cv2 as cv

from src.image_processor.image_processor import (DEFAULT_LEFT_LIFETIME,
                                                 find_stars,
                                                 prune_close_points,
                                                 star_tracker,
                                                 detect_blinking_star)


class DataGettersTestCase(unittest.TestCase):
    """ Class to test the image_processor script. """
    def test_find_stars_1(self):
        """ find_stars can detect one star. """

        image = cv.imread(str(Path("./data/images/stellarium-007.png")))
        if image is None:
            sys.exit("Could not read the image.")
        expected_result = [(17, 17)]

        result = find_stars(image)
        self.assertEqual(result, expected_result)

    def test_find_stars_2(self):
        """ find_stars can detect multiple stars. """

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

        result = find_stars(image)
        self.assertEqual(result, expected_result)

    def test_prune_close_points_1(self):
        """ prune_close_points can filter a duplicate star. """

        star_positions = [(8, 8), (10, 10)]
        expected_result = [(10, 10)]

        result = prune_close_points(star_positions, min_distance=5)
        self.assertEqual(result, expected_result)

    def test_prune_close_points_2(self):
        """ prune_close_points can ignore a distant star . """

        star_positions = [(8, 8), (10, 10), (15, 15)]
        expected_result = [(10, 10), (15, 15)]

        result = prune_close_points(star_positions, min_distance=5)
        self.assertEqual(result, expected_result)

    def test_star_tracker_1(self):
        """ star_tracker can add a new star. """

        fps = 60
        desired_blinking_freq = 30
        star_positions = [(10, 15)]
        detected_stars = {}
        expected_result = ({
            0: {
                "position": (10, 15),
                "times_detected": 1,
                "lifetime": 1,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps,
                "tickets_to_be_the_satellite": 0,
            }
        }, 1)

        result = star_tracker(star_positions,
                              detected_stars,
                              desired_blinking_freq,
                              fps,
                              next_star_id=0)
        self.assertEqual(result, expected_result)

    def test_star_tracker_2(self):
        """ star_tracker can keep track of a moving star. """

        fps = 60
        desired_blinking_freq = 30
        star_positions = [(11, 16)]
        detected_stars = {
            0: {
                "position": (10, 15),
                "times_detected": 1,
                "lifetime": 3,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps,
                "tickets_to_be_the_satellite": 0,
            }
        }
        expected_result = ({
            0: {
                "position": (11, 16),
                "times_detected": 2,
                "lifetime": 4,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps / 2,
                "tickets_to_be_the_satellite": 1,
            }
        }, 1)

        result = star_tracker(star_positions,
                              detected_stars,
                              desired_blinking_freq,
                              fps,
                              next_star_id=1)
        self.assertEqual(result, expected_result)

    def test_star_tracker_3(self):
        """ star_tracker can remember a hidden star. """

        fps = 60
        desired_blinking_freq = 30
        star_positions = []
        detected_stars = {
            0: {
                "position": (11, 16),
                "times_detected": 2,
                "lifetime": 2,
                "left_lifetime": DEFAULT_LEFT_LIFETIME,
                "blinking_freq": fps,
                "tickets_to_be_the_satellite": 0,
            }
        }
        expected_result = ({
            0: {
                "position": (11, 16),
                "times_detected": 2,
                "lifetime": 3,
                "left_lifetime": DEFAULT_LEFT_LIFETIME - 1,
                "blinking_freq": (2 / 3) * fps,
                "tickets_to_be_the_satellite": -2,
            }
        }, 1)

        result = star_tracker(star_positions,
                              detected_stars,
                              desired_blinking_freq,
                              fps,
                              next_star_id=1)
        self.assertEqual(result, expected_result)

    def test_star_tracker_4(self):
        """ star_tracker can forget a star not found for a long time. """

        fps = 60
        desired_blinking_freq = 30
        star_positions = []
        detected_stars = {0: {"position": (11, 16), "left_lifetime": 0}}
        expected_result = ({}, 1)

        result = star_tracker(star_positions,
                              detected_stars,
                              desired_blinking_freq,
                              fps,
                              next_star_id=1)
        self.assertEqual(result, expected_result)

    def test_detect_blinking_star(self):
        """ detect_blinking_star can detect the star that blinks at the desired
        blinking frequency. """

        detected_stars = {
            0: {
                "tickets_to_be_the_satellite": 6
            },
            1: {
                "tickets_to_be_the_satellite": 485
            },
            2: {
                "tickets_to_be_the_satellite": 8342
            },
            3: {
                "tickets_to_be_the_satellite": 141
            },
            4: {
                "tickets_to_be_the_satellite": 48
            }
        }
        expected_result = (2, {"tickets_to_be_the_satellite": 8342})

        result = detect_blinking_star(detected_stars)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
