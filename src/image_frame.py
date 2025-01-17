#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the ImageFrame class.
"""

from copy import deepcopy
from itertools import pairwise as itertools__pairwise

import cv2 as cv
from imutils import translate as imutils__translate
import numpy as np

from src.star import Star
from constants import (
    MARK_DETECTED_STARS,
    MARK_TRACKED_STARS,
    MARK_SHOOTING_STARS,
    MARK_SATELLITE,
    MARK_MOVEMENT_VECTOR,
    MARK_NEXT_EXPECTED_POSITION,
    MARK_LAST_PREDICTED_POSITION,
    COLORIZED_TRACKED_STARS,
    MARK_POSITION_DEFAULT_COLOR,
    MARK_SHOOTING_STAR_COLOR,
    MARK_SATELLITE_COLOR,
    MARK_NEXT_EXPECTED_POSITION_COLOR,
    MARK_LAST_PREDICTED_POSITION_COLOR,
    MARK_MOVEMENT_VECTOR_COLOR,
    MARK_RADIUS,
    MARK_THICKNESS,
)


class ImageFrame:
    def __init__(
        self,
        data: cv.mat_wrapper.Mat | np.ndarray,
        width: int = None,
        height: int = None,
        is_color: bool = None,
    ):
        self.data = data.copy()
        self.width = width if width is not None else data.shape[1]
        self.height = height if height is not None else data.shape[0]

        if is_color is not None:
            self.is_color = is_color
        else:
            self.is_color = len(data.shape) == 3 and data.shape[2] == 3

        self.frame_center = (self.width / 2, self.height / 2)

    def copy(self):
        """Method to return a deep copy of the current ImageFrame object."""

        return deepcopy(self)

    def to_grayscale(self) -> "ImageFrame":
        """Method to convert the current image frame to grayscale color space."""

        if self.is_color:
            return ImageFrame(
                cv.cvtColor(self.data, cv.COLOR_BGR2GRAY),
                width=self.width,
                height=self.height,
                is_color=False,
            )

        else:
            return self.copy()

    def to_colorspace(self) -> "ImageFrame":
        """Method to convert the current image frame to RGB color space."""

        if self.is_color:
            return self.copy()

        else:
            return ImageFrame(
                cv.cvtColor(self.data, cv.COLOR_GRAY2BGR),
                width=self.width,
                height=self.height,
                is_color=True,
            )

    def mark_position(
        self,
        target: tuple[int, int],
        color: tuple = MARK_POSITION_DEFAULT_COLOR,
        radius: int = MARK_RADIUS,
        thickness: int = MARK_THICKNESS,
    ):
        """Method to mark in the frame data a circle around the given position."""

        cv.circle(
            self.data,
            center=(
                int(target[0]),
                int(target[1]),
            ),
            color=color,
            radius=radius,
            thickness=thickness,
        )

    def mark_path(
        self,
        targets: Star | set[Star],
        color: tuple = MARK_MOVEMENT_VECTOR_COLOR,
        thickness: int = MARK_THICKNESS,
    ):
        """
        Method to mark in the frame data a line through the last detected positions of
        the given objects.
        """

        for target in {targets} if isinstance(targets, Star) else targets:
            last_positions = [
                [round(axis) for axis in pos]
                for pos in target.last_positions
                if pos is not None
            ]
            for pos_1, pos_2 in itertools__pairwise(last_positions):
                cv.line(
                    self.data,
                    pt1=pos_1,
                    pt2=pos_2,
                    color=color,
                    thickness=thickness,
                )

    def mark(
        self,
        new_star_positions: list[tuple[int, int]] = None,
        tracked_stars: set[Star] = None,
        shooting_stars: set[Star] = None,
        satellite: Star = None,
        mark_new_stars: bool = MARK_DETECTED_STARS,
        mark_tracked_stars: bool = MARK_TRACKED_STARS,
        mark_shooting_stars: bool = MARK_SHOOTING_STARS,
        mark_satellite: bool = MARK_SATELLITE,
        mark_movement_vector: bool = MARK_MOVEMENT_VECTOR,
        mark_next_expected_position: bool = MARK_NEXT_EXPECTED_POSITION,
        mark_last_predicted_position: bool = MARK_LAST_PREDICTED_POSITION,
        colorized_tracked_stars: bool = COLORIZED_TRACKED_STARS,
        mark_position_default_color: tuple = MARK_POSITION_DEFAULT_COLOR,
        mark_shooting_star_color: tuple = MARK_SHOOTING_STAR_COLOR,
        mark_satellite_color: tuple = MARK_SATELLITE_COLOR,
        mark_next_expected_position_color: tuple = MARK_NEXT_EXPECTED_POSITION_COLOR,
        mark_last_predicted_position_color: tuple = MARK_LAST_PREDICTED_POSITION_COLOR,
        mark_radius: int = MARK_RADIUS,
        mark_thickness: int = MARK_THICKNESS,
    ):
        """Method to mark information about the detected elements in the frame data."""

        if mark_new_stars and new_star_positions is not None:
            for star in new_star_positions:
                self.mark_position(
                    star,
                    color=mark_position_default_color,
                    radius=mark_radius,
                    thickness=mark_thickness,
                )

        if mark_tracked_stars and tracked_stars is not None:
            for star in tracked_stars:
                if colorized_tracked_stars and isinstance(star, Star):
                    self.mark_position(
                        star.last_detected_position,
                        color=star.color,
                        radius=mark_radius,
                        thickness=mark_thickness,
                    )
                else:
                    self.mark_position(
                        star.last_detected_position,
                        color=mark_position_default_color,
                        radius=mark_radius,
                        thickness=mark_thickness,
                    )

        if mark_shooting_stars and shooting_stars is not None:
            for star in shooting_stars:
                if mark_movement_vector:
                    self.mark_path(star)

                self.mark_position(
                    star.last_detected_position,
                    color=mark_shooting_star_color,
                    radius=mark_radius,
                    thickness=mark_thickness,
                )

                if mark_next_expected_position:
                    self.mark_position(
                        star.next_expected_position,
                        color=mark_next_expected_position_color,
                        radius=mark_radius,
                        thickness=mark_thickness,
                    )
                if mark_last_predicted_position:
                    self.mark_position(
                        star.last_predicted_position,
                        color=mark_last_predicted_position_color,
                        radius=mark_radius,
                        thickness=mark_thickness,
                    )

        if mark_satellite and satellite is not None:
            if mark_movement_vector:
                self.mark_path(satellite)

            self.mark_position(
                satellite.last_detected_position,
                color=mark_satellite_color,
                radius=mark_radius,
                thickness=mark_thickness,
            )

            if mark_next_expected_position:
                self.mark_position(
                    satellite.next_expected_position,
                    color=mark_next_expected_position_color,
                    radius=mark_radius,
                    thickness=mark_thickness,
                )
            if mark_last_predicted_position:
                self.mark_position(
                    satellite.last_predicted_position,
                    color=mark_last_predicted_position_color,
                    radius=mark_radius,
                    thickness=mark_thickness,
                )

    def tracking_phase_video_simulation(self, satellite: Star):
        """
        Method to simulate the tracking phase of the satellite by moving the frame data
        to set the target satellite in the center of the video.
        </br></br>

        Caution: This method does not modify the relative positions of the detected
        objects in the frame, later modifications should be done to keep the objects
        in the correct positions if required.
        """

        translation_vector = [
            self.frame_center[0] - satellite.next_expected_position[0],
            self.frame_center[1] - satellite.next_expected_position[1],
        ]

        self.data = imutils__translate(
            self.data,
            translation_vector[0],
            translation_vector[1],
        )
