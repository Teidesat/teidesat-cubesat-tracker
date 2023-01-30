#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TeideSat Satellite Tracking for the Optical Ground Station

# ToDo: Check if this is correct
#     This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#     This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
# have received a copy of the GNU General Public License along with this program. If 
# not, see <https://www.gnu.org/licenses/>.
"""  # ToDo: complete module docstring

# ToDo: add the missing module information
__authors__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
# __contact__ = "mail@example.com"
# __copyright__ = "Copyright $YEAR, $COMPANY_NAME"
__credits__ = ["Jorge Sierra", "Sergio Tabares Hernández"]
__date__ = "2022/06/12"
__deprecated__ = False
# __email__ = "mail@example.com"
# __license__ = "GPLv3"
__maintainer__ = "Sergio Tabares Hernández"
__status__ = "Production"
__version__ = "0.0.8"

from copy import deepcopy
from itertools import pairwise
import sys
from time import perf_counter

import cv2 as cv

from src.image_processor import (
    detect_stars,
    track_stars,
    detect_blinking_star,
    detect_shooting_stars,
)
from src.star import Star
from constants import (
    STAR_DETECTOR_THRESHOLD,
    FAST,
    MIN_PRUNE_DISTANCE,
    SAT_DESIRED_BLINKING_FREQ,
    MOVEMENT_THRESHOLD,
    PX_SENSITIVITY,
    PATH_VIDEO,
    PATH_SAT_LOG,
    RGB_IMAGE,
    VIDEO_FROM_CAMERA,
    SHOW_VIDEO_RESULT,
    COLORIZED_TRACKED_STARS,
    OUTPUT_VIDEO_TO_FILE,
    PATH_OUTPUT_VIDEO,
)


def main():
    """Main function to start the program execution."""

    if VIDEO_FROM_CAMERA:
        video_path = 0  # Default webcam id
        print("Processing video from camera number ", video_path)

    else:
        video_path = str(PATH_VIDEO)
        print("Processing video from:", video_path)

    vid_cap = cv.VideoCapture(video_path)
    if not vid_cap.isOpened():
        sys.exit("\nError: Unable to open video.")

    satellite_detection_test(vid_cap)


def satellite_detection_test(
    vid_cap,
    sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
    star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
    fast: bool = FAST,
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
    movement_threshold: float = MOVEMENT_THRESHOLD,
    rgb_image: bool = RGB_IMAGE,
    show_video_result: bool = SHOW_VIDEO_RESULT,
    output_video_to_file: bool = OUTPUT_VIDEO_TO_FILE,
):
    """Function to detect and track the satellite."""

    # if VIDEO_FROM_CAMERA this could not work
    video_fps = vid_cap.get(cv.CAP_PROP_FPS)

    star_detector = cv.FastFeatureDetector_create(threshold=star_detector_threshold)

    tracked_stars: set[Star] = set()
    satellite_log: list[Star] = []

    wait_time = 1
    wait_options = {
        ord("z"): 1,
        ord("x"): 100,
        ord("c"): 1000,
        ord("v"): 0,
    }

    processed_frames = 0
    start_time = perf_counter()

    output_video = create_export_video_file(vid_cap) if output_video_to_file else None

    if show_video_result:
        cv.namedWindow("Satellite detection", cv.WINDOW_NORMAL)

    while True:
        success, frame = vid_cap.read()
        if not success:
            break

        processed_frames += 1

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) if rgb_image else frame

        new_star_positions = detect_stars(gray, star_detector, fast, min_prune_distance)

        track_stars(
            new_star_positions,
            tracked_stars,
            sat_desired_blinking_freq,
            video_fps,
        )

        shooting_stars = detect_shooting_stars(tracked_stars, movement_threshold)

        satellite = detect_blinking_star(shooting_stars)

        show_frame = (
            frame.copy() if rgb_image else cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        )

        # draw_found_stars(show_frame, new_star_positions)
        draw_tracked_stars(show_frame, tracked_stars)
        draw_shooting_stars(show_frame, shooting_stars)
        draw_path(show_frame, shooting_stars)

        if satellite is not None:
            satellite_log.append(deepcopy(satellite))
            draw_satellite(show_frame, satellite)
            draw_path(show_frame, satellite)

        if output_video_to_file:
            output_video.write(show_frame)

        if show_video_result:
            cv.imshow("Satellite detection", show_frame)

            key = cv.waitKey(wait_time)

            if key == ord("q"):
                break

            wait_time = wait_options.get(key, wait_time)

    print_time_statistics(processed_frames, start_time)
    export_satellite_log(satellite_log)

    if output_video_to_file:
        print(f"Video saved on '{str(PATH_OUTPUT_VIDEO)}'")

    cv.destroyAllWindows()


def print_time_statistics(
    processed_frames: int,
    start_time: float,
):
    """Function to print processing time statistics."""

    processing_time = perf_counter() - start_time

    print("  Processed frames:", processed_frames)
    print("  Time needed:", processing_time)
    print("  FPS:", processed_frames / processing_time)
    print()


def draw_found_stars(
    show_frame,
    found_stars: list[tuple[int, int]],
    radius: int = PX_SENSITIVITY,
    color: tuple = (0, 0, 100),
    thickness: int = 2,
):
    """
    Function to draw in the given frame a circle around every found star.

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    for star in found_stars:
        cv.circle(
            show_frame,
            center=(
                int(star[0]),
                int(star[1]),
            ),
            radius=radius,
            color=color,
            thickness=thickness,
        )


def draw_tracked_stars(
    show_frame,
    tracked_stars: set[Star],
    radius: int = PX_SENSITIVITY,
    color: tuple = None,
    thickness: int = 2,
    colorized_tracked_stars: bool = COLORIZED_TRACKED_STARS,
):
    """
    Function to draw in the given frame a circle around every tracked star.

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    for star in tracked_stars:

        if color is None:
            if colorized_tracked_stars:
                draw_color = star.color
            else:
                draw_color = (0, 0, 100)
        else:
            draw_color = color

        cv.circle(
            show_frame,
            center=(
                int(star.last_positions[-1][0]),
                int(star.last_positions[-1][1]),
            ),
            radius=radius,
            color=draw_color,
            thickness=thickness,
        )


def draw_shooting_stars(
    show_frame,
    shooting_stars: set[Star],
    radius: int = PX_SENSITIVITY,
    color: tuple = (0, 200, 200),
    thickness: int = 2,
):
    """
    Function to draw in the given frame a circle around every shooting star.

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    for star in shooting_stars:
        cv.circle(
            show_frame,
            center=(
                int(star.last_positions[-1][0]),
                int(star.last_positions[-1][1]),
            ),
            radius=radius,
            color=color,
            thickness=thickness,
        )


def draw_satellite(
    show_frame,
    satellite: Star,
    radius: int = PX_SENSITIVITY,
    color: tuple = (0, 200, 0),
    thickness: int = 2,
):
    """
    Function to draw in the given frame a circle around the satellite detected.

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    cv.circle(
        show_frame,
        center=(
            int(satellite.last_positions[-1][0]),
            int(satellite.last_positions[-1][1]),
        ),
        radius=radius,
        color=color,
        thickness=thickness,
    )


def draw_path(
    show_frame,
    targets,
    color: tuple = (200, 200, 0),
    thickness: int = 1,
):
    """
    Function to draw in the given frame a line through the last detected
    positions of the given objects.

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    for target in {targets} if isinstance(targets, Star) else targets:
        last_positions = [
            [round(axis) for axis in pos] for pos in target.last_positions
        ]
        for pos_1, pos_2 in pairwise(last_positions):
            cv.line(
                show_frame,
                pt1=pos_1,
                pt2=pos_2,
                color=color,
                thickness=thickness,
            )


def create_export_video_file(vid_cap):
    """Function to create a video file to export the processed frames."""

    width = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid_cap.get(cv.CAP_PROP_FPS))
    frame_count = int(vid_cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo format: {width}x{height} px - {fps} fps")
    print("Number of frames:", frame_count)

    output_video = cv.VideoWriter(
        str(PATH_OUTPUT_VIDEO),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not output_video.isOpened():
        sys.exit("\nError: Unable to create video file.")

    return output_video


def export_satellite_log(satellite_log: list[Star]):
    """Function to export the satellite log into a file."""

    with open(str(PATH_SAT_LOG), "w", encoding="utf-8-sig") as file:
        print(
            "id;",
            "last_times_detected;",
            "lifetime;",
            "left_lifetime;",
            "detection_confidence;",
            "blinking_freq;",
            "movement_vector;",
            "last_positions;",
            file=file,
        )

        for star in satellite_log:
            print(
                f"{star.id};",
                f"{star.last_times_detected};",
                f"{star.lifetime};",
                f"{star.left_lifetime};",
                f"{star.detection_confidence};",
                f"{star.blinking_freq};",
                f"{star.movement_vector};",
                f"{star.last_positions};",
                file=file,
            )


if __name__ == "__main__":
    main()
