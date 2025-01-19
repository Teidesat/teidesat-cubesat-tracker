#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the image processing functions, star detection and star
 tracking algorithms.
"""

from math import dist
from typing import Optional

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.image_frame import ImageFrame
from src.star import Star

from constants import (
    VIDEO_FPS,
    STAR_DETECTION_MODE,
    STAR_DETECTOR_THRESHOLD,
    NEURAL_NETWORK_MODEL_PATH,
    PRUNE_CLOSE_POINTS,
    MIN_PRUNE_DISTANCE,
    DEFAULT_LEFT_LIFETIME,
    SAT_DESIRED_BLINKING_FREQ,
    MIN_DETECTION_CONFIDENCE,
    DEFAULT_VECTOR,
    MOVEMENT_THRESHOLD,
)


def load_star_detector(
    star_detection_mode: str = STAR_DETECTION_MODE,
    star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
    neural_network_model_path: str = NEURAL_NETWORK_MODEL_PATH,
):

    if star_detection_mode == "OPEN_CV":
        star_detector = cv.FastFeatureDetector_create(threshold=star_detector_threshold)

    elif star_detection_mode == "NEURAL_NETWORK":
        star_detector = keras.models.load_model(neural_network_model_path)

    else:
        raise ValueError(f"Invalid star detector mode: {star_detection_mode}")

    return star_detector


def detect_sky_objects(input_image, model, input_tensor_shape=(256, 256)):
    """Function to apply the model inference over the given image."""

    # Adapt the input image to the model input shape and format
    resized_image = cv.resize(input_image, dsize=input_tensor_shape)
    input_tensor = tf.convert_to_tensor([resized_image])

    # Apply the model to the input image
    predicted_mask = model.predict(input_tensor)[0]

    # Reverse the one-hot encoding to create a single segmentation mask
    segmentation_mask = np.argmax(predicted_mask, axis=-1)

    # Resize the mask to the original image size
    result_mask = cv.resize(
        segmentation_mask.astype(np.uint8),
        dsize=(input_image.shape[1], input_image.shape[0]),
    )

    return result_mask


def get_sky_objects_positions(segmentation_mask):
    """
    Function to get the central positions of the detected objects at the given
    segmentation mask.
    """

    # Get the contours of each detected region
    contours, _ = cv.findContours(
        segmentation_mask,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE,
    )

    min_detected_object_size = 10
    sky_objects_positions = []
    for contour in contours:
        if cv.contourArea(contour) < min_detected_object_size:
            continue

        # Get the bounding box of each contour
        corner_x_coord, corner_y_coord, width, height = cv.boundingRect(contour)

        # Get the center of the bounding box
        center_x_coord = corner_x_coord + width / 2
        center_y_coord = corner_y_coord + height / 2

        # Add the center coordinates to the list of detected objects positions
        sky_objects_positions.append((center_x_coord, center_y_coord))

    return sky_objects_positions


def detect_stars(
    image_frame: ImageFrame,
    star_detector,
    star_detection_mode: str = STAR_DETECTION_MODE,
    prune_close_points: bool = PRUNE_CLOSE_POINTS,
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
) -> list[tuple[int, int]]:
    """Function to get all the bright points of a given image."""

    if star_detection_mode == "OPEN_CV":
        keypoints = star_detector.detect(image_frame.data, None)
        points = [keypoint.pt for keypoint in keypoints]

    elif star_detection_mode == "NEURAL_NETWORK":
        inference_result_mask = detect_sky_objects(
            image_frame.data, star_detector
        ).astype(np.uint8)
        points = get_sky_objects_positions(inference_result_mask)

    else:
        raise ValueError(f"Invalid star detector mode: {star_detection_mode}")

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
