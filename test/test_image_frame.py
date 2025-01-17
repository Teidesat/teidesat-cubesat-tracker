#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program tests the correct functionality of the functions at src/image_frame.py
 file.
"""


import unittest

import cv2 as cv
import numpy as np

from src.image_frame import (
    ImageFrame,
    MARK_POSITION_DEFAULT_COLOR,
    MARK_SHOOTING_STAR_COLOR,
    MARK_SATELLITE_COLOR,
    MARK_NEXT_EXPECTED_POSITION_COLOR,
    MARK_LAST_PREDICTED_POSITION_COLOR,
    MARK_MOVEMENT_VECTOR_COLOR,
    MARK_RADIUS,
    MARK_THICKNESS,
)
from src.star import Star


class ImageFrameClassTestCase(unittest.TestCase):
    """Class to test the ImageFrame class and its methods."""

    def test_simple_image_frame_1(self):
        """A simple image frame can be created only with basic frame data."""

        data = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ],
        )

        image_frame = ImageFrame(data)

        self.assertTrue(np.array_equal(data, image_frame.data))
        self.assertEqual(image_frame.width, 2)
        self.assertEqual(image_frame.height, 3)
        self.assertEqual(image_frame.is_color, False)

    def test_simple_image_frame_2(self):
        """A simple image frame can be created with optional arguments."""

        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        image_frame = ImageFrame(
            data,
            width=data.shape[1],
            height=data.shape[0],
            is_color=(len(data.shape) == 3 and data.shape[2] == 3),
        )

        self.assertTrue(np.array_equal(data, image_frame.data))
        self.assertEqual(image_frame.width, 3)
        self.assertEqual(image_frame.height, 2)
        self.assertEqual(image_frame.is_color, False)

    def test_real_image_frame_1(self):
        """
        A color image frame can be created from a real color image only with basic frame
        data.
        """

        image_data = cv.imread("./data/images/original.jpg")
        image_frame = ImageFrame(image_data)

        self.assertTrue(np.array_equal(image_data, image_frame.data))
        self.assertEqual(image_frame.width, 1280)
        self.assertEqual(image_frame.height, 720)
        self.assertEqual(image_frame.is_color, True)

    def test_real_image_frame_2(self):
        """
        A grayscale image frame can be created from a real grayscale image only with
        basic frame data.
        """

        grayscale_image_data = cv.cvtColor(
            cv.imread("./data/images/original.jpg"), cv.COLOR_BGR2GRAY
        )
        image_frame = ImageFrame(grayscale_image_data)

        self.assertTrue(np.array_equal(grayscale_image_data, image_frame.data))
        self.assertEqual(image_frame.width, 1280)
        self.assertEqual(image_frame.height, 720)
        self.assertEqual(image_frame.is_color, False)

    def test_real_image_frame_3(self):
        """An image frame can be created from a real image with optional arguments."""

        image_data = cv.imread("./data/images/original.jpg")
        image_frame = ImageFrame(
            image_data,
            width=image_data.shape[1],
            height=image_data.shape[0],
            is_color=(len(image_data.shape) == 3 and image_data.shape[2] == 3),
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))
        self.assertEqual(image_frame.width, 1280)
        self.assertEqual(image_frame.height, 720)
        self.assertEqual(image_frame.is_color, True)

    def test_copy(self):
        """A copy of an image frame will return a new image frame with the same data."""

        original_image_frame = ImageFrame(cv.imread("./data/images/original.jpg"))
        copied_image_frame = original_image_frame.copy()

        self.assertTrue(
            np.array_equal(original_image_frame.data, copied_image_frame.data)
        )
        self.assertEqual(original_image_frame.width, copied_image_frame.width)
        self.assertEqual(original_image_frame.height, copied_image_frame.height)
        self.assertEqual(original_image_frame.is_color, copied_image_frame.is_color)

    def test_to_grayscale_1(self):
        """A color image frame can be converted to grayscale."""

        image_data = cv.imread("./data/images/original.jpg")
        image_frame = ImageFrame(image_data)
        grayscale_image_frame = image_frame.to_grayscale()

        self.assertTrue(
            np.array_equal(
                cv.cvtColor(image_data, cv.COLOR_BGR2GRAY), grayscale_image_frame.data
            )
        )
        self.assertEqual(grayscale_image_frame.width, 1280)
        self.assertEqual(grayscale_image_frame.height, 720)
        self.assertEqual(grayscale_image_frame.is_color, False)

    def test_to_grayscale_2(self):
        """
        A grayscale image frame will return a copy of itself when converted to
        grayscale.
        """

        grayscale_image_data = cv.cvtColor(
            cv.imread("./data/images/original.jpg"), cv.COLOR_BGR2GRAY
        )
        image_frame = ImageFrame(grayscale_image_data)
        grayscale_image_frame = image_frame.to_grayscale()

        self.assertTrue(
            np.array_equal(grayscale_image_data, grayscale_image_frame.data)
        )
        self.assertEqual(grayscale_image_frame.width, 1280)
        self.assertEqual(grayscale_image_frame.height, 720)
        self.assertEqual(grayscale_image_frame.is_color, False)

    def test_to_colorspace_1(self):
        """A grayscale image frame can be converted to a color image frame."""

        grayscale_image_data = cv.cvtColor(
            cv.imread("./data/images/original.jpg"), cv.COLOR_BGR2GRAY
        )
        image_frame = ImageFrame(grayscale_image_data)
        color_image_frame = image_frame.to_colorspace()

        self.assertTrue(
            np.array_equal(
                cv.cvtColor(grayscale_image_data, cv.COLOR_GRAY2BGR),
                color_image_frame.data,
            )
        )
        self.assertEqual(color_image_frame.width, 1280)
        self.assertEqual(color_image_frame.height, 720)
        self.assertEqual(color_image_frame.is_color, True)

    def test_to_colorspace_2(self):
        """
        A color image frame will return a copy of itself when converted to colorspace.
        """

        image_data = cv.imread("./data/images/original.jpg")
        image_frame = ImageFrame(image_data)
        color_image_frame = image_frame.to_colorspace()

        self.assertTrue(np.array_equal(image_data, color_image_frame.data))
        self.assertEqual(color_image_frame.width, 1280)
        self.assertEqual(color_image_frame.height, 720)
        self.assertEqual(color_image_frame.is_color, True)

    def test_mark_position_1(self):
        """
        A circle can be drawn around a point in an image frame with default values.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (25, 25)

        image_frame.mark_position(target)

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=MARK_POSITION_DEFAULT_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_position_2(self):
        """A circle can be drawn around a point in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (25, 25)
        color = (0, 0, 255)
        radius = 10
        thickness = 3

        image_frame.mark_position(
            target=target,
            color=color,
            radius=radius,
            thickness=thickness,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_path_1(self):
        """A line can be drawn in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (10, 15)
        end = (40, 35)

        star = Star(last_positions=[start, end])
        image_frame.mark_path(star)

        cv.line(
            image_data,
            pt1=start,
            pt2=end,
            color=MARK_MOVEMENT_VECTOR_COLOR,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_path_2(self):
        """A line can be drawn in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (10, 15)
        end = (40, 35)
        color = (0, 255, 0)
        thickness = 3

        star = Star(last_positions=[start, end])
        image_frame.mark_path(star, color=color, thickness=thickness)

        cv.line(
            image_data,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_path_3(self):
        """
        A line can be drawn in an image frame for a star with multiple last positions.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        positions = [(10, 15), (20, 35), (30, 25), (40, 35)]

        star = Star(last_positions=positions)
        image_frame.mark_path(star)

        for start, end in zip(positions[:-1], positions[1:]):
            cv.line(
                image_data,
                pt1=start,
                pt2=end,
                color=MARK_MOVEMENT_VECTOR_COLOR,
                thickness=MARK_THICKNESS,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_path_4(self):
        """
        Multiple lines can be drawn in an image frame for multiple stars with different
        last positions.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        positions_1 = [(10, 15), (20, 35), (30, 25), (40, 35)]
        positions_2 = [(20, 25), (30, 45), (40, 35), (10, 45)]

        stars = {
            Star(last_positions=positions_1),
            Star(last_positions=positions_2),
        }

        image_frame.mark_path(stars)

        for star in stars:
            for start, end in zip(star.last_positions[:-1], star.last_positions[1:]):
                cv.line(
                    image_data,
                    pt1=start,
                    pt2=end,
                    color=MARK_MOVEMENT_VECTOR_COLOR,
                    thickness=MARK_THICKNESS,
                )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_new_stars_1(self):
        """A new star position can be marked in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (25, 25)

        image_frame.mark(
            new_star_positions=[target],
            mark_new_stars=True,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=MARK_POSITION_DEFAULT_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_new_stars_2(self):
        """A new star position can be marked in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (25, 25)
        color = (0, 0, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            new_star_positions=[target],
            mark_new_stars=True,
            mark_position_default_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_new_stars_3(self):
        """
        Multiple new star positions can be marked in an image frame with default values.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(25, 25), (30, 30), (35, 35)]

        image_frame.mark(
            new_star_positions=targets,
            mark_new_stars=True,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=MARK_POSITION_DEFAULT_COLOR,
                radius=MARK_RADIUS,
                thickness=MARK_THICKNESS,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_new_stars_4(self):
        """
        Multiple new star positions can be marked in an image frame with custom values.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(25, 25), (30, 30), (35, 35)]
        color = (0, 0, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            new_star_positions=targets,
            mark_new_stars=True,
            mark_position_default_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=color,
                radius=radius,
                thickness=thickness,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_tracked_stars_1(self):
        """A tracked star can be marked in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (30, 30)

        image_frame.mark(
            tracked_stars={Star(last_positions=[target])},
            mark_tracked_stars=True,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=MARK_POSITION_DEFAULT_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_tracked_stars_2(self):
        """A tracked star can be marked in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (30, 30)
        color = (0, 0, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            tracked_stars={Star(last_positions=[target])},
            mark_tracked_stars=True,
            mark_position_default_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_tracked_stars_3(self):
        """Multiple tracked stars can be marked in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(30, 35), (40, 40), (45, 35)]

        image_frame.mark(
            tracked_stars={Star(last_positions=[target]) for target in targets},
            mark_tracked_stars=True,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=MARK_POSITION_DEFAULT_COLOR,
                radius=MARK_RADIUS,
                thickness=MARK_THICKNESS,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_tracked_stars_4(self):
        """Multiple tracked stars can be marked in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(30, 35), (40, 40), (45, 35)]
        color = (0, 0, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            tracked_stars={Star(last_positions=[target]) for target in targets},
            mark_tracked_stars=True,
            mark_position_default_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=color,
                radius=radius,
                thickness=thickness,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_tracked_stars_5(self):
        """Multiple tracked stars can be marked in an image frame with unique colors."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(30, 35), (40, 40), (45, 35)]
        colors = [(0, 0, 100), (0, 100, 0), (100, 0, 0)]
        radius = 5
        thickness = 3

        image_frame.mark(
            tracked_stars={
                Star(last_positions=[target], color=list(color))
                for target, color in zip(targets, colors)
            },
            mark_tracked_stars=True,
            colorized_tracked_stars=True,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        for target, color in zip(targets, colors):
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=color,
                radius=radius,
                thickness=thickness,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_1(self):
        """A shooting star can be marked in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (35, 35)

        image_frame.mark(
            shooting_stars={Star(last_positions=[target])},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=MARK_SHOOTING_STAR_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_2(self):
        """A shooting star can be marked in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (35, 35)
        color = (0, 100, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            shooting_stars={Star(last_positions=[target])},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
            mark_shooting_star_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_3(self):
        """
        Multiple shooting stars can be marked in an image frame with default values.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(35, 35), (40, 40), (45, 45)]

        image_frame.mark(
            shooting_stars={Star(last_positions=[target]) for target in targets},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=MARK_SHOOTING_STAR_COLOR,
                radius=MARK_RADIUS,
                thickness=MARK_THICKNESS,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_4(self):
        """
        Multiple shooting stars can be marked in an image frame with custom values.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        targets = [(35, 35), (40, 40), (45, 45)]
        color = (0, 100, 100)
        radius = 5
        thickness = 3

        image_frame.mark(
            shooting_stars={Star(last_positions=[target]) for target in targets},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
            mark_shooting_star_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        for target in targets:
            cv.circle(
                image_data,
                center=(
                    int(target[0]),
                    int(target[1]),
                ),
                color=color,
                radius=radius,
                thickness=thickness,
            )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_5(self):
        """A shooting star can be marked in an image frame with its movement vector."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (35, 35)
        end = (40, 40)

        image_frame.mark(
            shooting_stars={Star(last_positions=[start, end])},
            mark_shooting_stars=True,
            mark_movement_vector=True,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
        )

        cv.line(
            image_data,
            pt1=start,
            pt2=end,
            color=MARK_MOVEMENT_VECTOR_COLOR,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_SHOOTING_STAR_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_6(self):
        """
        A shooting star can be marked in an image frame with its next expected position.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (35, 35)
        end = (40, 40)

        image_frame.mark(
            shooting_stars={Star(last_positions=[start], next_expected_position=end)},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=True,
            mark_last_predicted_position=False,
        )

        cv.circle(
            image_data,
            center=(
                int(start[0]),
                int(start[1]),
            ),
            color=MARK_SHOOTING_STAR_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_NEXT_EXPECTED_POSITION_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_shooting_stars_7(self):
        """
        A shooting star can be marked in an image frame with its last predicted position.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (35, 35)
        end = (40, 40)

        image_frame.mark(
            shooting_stars={Star(last_positions=[start], last_predicted_position=end)},
            mark_shooting_stars=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=True,
        )

        cv.circle(
            image_data,
            center=(
                int(start[0]),
                int(start[1]),
            ),
            color=MARK_SHOOTING_STAR_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_LAST_PREDICTED_POSITION_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_satellite_1(self):
        """A satellite can be marked in an image frame with default values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (40, 40)

        image_frame.mark(
            satellite=Star(last_positions=[target]),
            mark_satellite=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=MARK_SATELLITE_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_satellite_2(self):
        """A satellite can be marked in an image frame with custom values."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        target = (40, 40)
        color = (0, 100, 0)
        radius = 5
        thickness = 3

        image_frame.mark(
            satellite=Star(last_positions=[target]),
            mark_satellite=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
            mark_satellite_color=color,
            mark_radius=radius,
            mark_thickness=thickness,
        )

        cv.circle(
            image_data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_satellite_3(self):
        """A satellite can be marked in an image frame with its movement vector."""

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (40, 40)
        end = (45, 45)

        image_frame.mark(
            satellite=Star(last_positions=[start, end]),
            mark_satellite=True,
            mark_movement_vector=True,
            mark_next_expected_position=False,
            mark_last_predicted_position=False,
        )

        cv.line(
            image_data,
            pt1=start,
            pt2=end,
            color=MARK_MOVEMENT_VECTOR_COLOR,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_SATELLITE_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_satellite_4(self):
        """
        A satellite can be marked in an image frame with its next expected position.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (40, 40)
        end = (45, 45)

        image_frame.mark(
            satellite=Star(last_positions=[start], next_expected_position=end),
            mark_satellite=True,
            mark_movement_vector=False,
            mark_next_expected_position=True,
            mark_last_predicted_position=False,
        )

        cv.circle(
            image_data,
            center=(
                int(start[0]),
                int(start[1]),
            ),
            color=MARK_SATELLITE_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_NEXT_EXPECTED_POSITION_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    def test_mark_satellite_5(self):
        """
        A satellite can be marked in an image frame with its last predicted position.
        """

        image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        image_frame = ImageFrame(image_data)

        start = (40, 40)
        end = (45, 45)

        image_frame.mark(
            satellite=Star(last_positions=[start], last_predicted_position=end),
            mark_satellite=True,
            mark_movement_vector=False,
            mark_next_expected_position=False,
            mark_last_predicted_position=True,
        )

        cv.circle(
            image_data,
            center=(
                int(start[0]),
                int(start[1]),
            ),
            color=MARK_SATELLITE_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        cv.circle(
            image_data,
            center=(
                int(end[0]),
                int(end[1]),
            ),
            color=MARK_LAST_PREDICTED_POSITION_COLOR,
            radius=MARK_RADIUS,
            thickness=MARK_THICKNESS,
        )

        self.assertTrue(np.array_equal(image_data, image_frame.data))

    # ToDo: Implement unit tests for the 'tracking_phase_video_simulation' method.
