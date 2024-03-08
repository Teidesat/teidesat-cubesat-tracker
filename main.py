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
__version__ = "0.0.9"

from copy import deepcopy
from itertools import pairwise
import sys
from time import perf_counter

import cv2 as cv
from imutils import translate

from src.image_processor import (
    detect_stars,
    track_stars,
    detect_blinking_star,
    detect_shooting_stars,
)
from src.star import Star
from constants import (
    VIDEO_FROM_CAMERA,
    CAMERA_INDEX,
    PATH_INPUT_VIDEO,
    RGB_IMAGE,
    SHOW_VIDEO_RESULT,
    SIMULATE_TRACKING,
    MARK_DETECTED_STARS,
    MARK_TRACKED_STARS,
    MARK_SHOOTING_STARS,
    MARK_SATELLITE,
    MARK_MOVEMENT_VECTOR,
    MARK_NEXT_EXPECTED_POSITION,
    MARK_LAST_PREDICTED_POSITION,
    COLORIZED_TRACKED_STARS,
    PX_SENSITIVITY,
    OUTPUT_RAW_VIDEO_TO_FILE,
    PATH_OUTPUT_RAW_VIDEO,
    OUTPUT_PROCESSED_VIDEO_TO_FILE,
    PATH_OUTPUT_PROCESSED_VIDEO,
    OUTPUT_SAT_LOG_TO_FILE,
    PATH_OUTPUT_SAT_LOG,
    STAR_DETECTOR_THRESHOLD,
    PRUNE_CLOSE_POINTS,
    MIN_PRUNE_DISTANCE,
    SAT_DESIRED_BLINKING_FREQ,
    MOVEMENT_THRESHOLD,
)


def main():
    """Main function to start the program execution."""

    if VIDEO_FROM_CAMERA:
        video_path = CAMERA_INDEX
        print("Processing video from camera number ", video_path)

    else:
        video_path = str(PATH_INPUT_VIDEO)
        print("Processing video from:", video_path)

    vid_cap = cv.VideoCapture(video_path)
    if not vid_cap.isOpened():
        sys.exit("\nError: Unable to open video.")

    satellite_detection_test(vid_cap)


def satellite_detection_test(
    vid_cap,
    sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
    star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
    prune_close_points: bool = PRUNE_CLOSE_POINTS,
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
    movement_threshold: float = MOVEMENT_THRESHOLD,
    rgb_image: bool = RGB_IMAGE,
    show_video_result: bool = SHOW_VIDEO_RESULT,
    output_sat_log_to_file: bool = OUTPUT_SAT_LOG_TO_FILE,
    path_output_sat_log: str = str(PATH_OUTPUT_SAT_LOG),
    output_raw_video_to_file: bool = OUTPUT_RAW_VIDEO_TO_FILE,
    path_output_raw_video: str = str(PATH_OUTPUT_RAW_VIDEO),
    output_processed_video_to_file: bool = OUTPUT_PROCESSED_VIDEO_TO_FILE,
    path_output_processed_video: str = str(PATH_OUTPUT_PROCESSED_VIDEO),
):
    """Function to detect and track the satellite."""

    # if VIDEO_FROM_CAMERA this could not work
    video_fps = vid_cap.get(cv.CAP_PROP_FPS)

    if SIMULATE_TRACKING:
        frame_width = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_center = (frame_width / 2, frame_height / 2)
    else:
        frame_center = None

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

    output_raw_video = (
        create_export_video_file(vid_cap, path_output_raw_video)
        if output_raw_video_to_file
        else None
    )

    output_processed_video = (
        create_export_video_file(vid_cap, path_output_processed_video)
        if output_processed_video_to_file
        else None
    )

    if show_video_result:
        cv.namedWindow("Satellite detection", cv.WINDOW_NORMAL)

    while True:
        success, frame = vid_cap.read()
        if not success:
            break

        processed_frames += 1

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) if rgb_image else frame

        new_star_positions = detect_stars(
            gray,
            star_detector,
            prune_close_points,
            min_prune_distance,
        )

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

        draw_in_frame(
            show_frame,
            new_star_positions,
            tracked_stars,
            shooting_stars,
            satellite,
        )

        if satellite is not None:
            satellite_log.append(deepcopy(satellite))

            if frame_center is not None:
                show_frame = tracking_phase_video_simulation(
                    satellite, show_frame, frame_center
                )

        if output_raw_video is not None:
            output_raw_video.write(frame)

        if output_processed_video is not None:
            output_processed_video.write(show_frame)

        if show_video_result:
            cv.imshow("Satellite detection", show_frame)

            key = cv.waitKey(wait_time)

            if key == ord("q"):
                break

            wait_time = wait_options.get(key, wait_time)

    print_time_statistics(processed_frames, start_time)

    if output_sat_log_to_file:
        export_satellite_log(satellite_log, path_output_sat_log)

    if output_raw_video is not None:
        print(f"Raw video saved on '{path_output_raw_video}'")

    if output_processed_video is not None:
        print(f"Processed video saved on '{path_output_processed_video}'")

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


def draw_in_frame(
    show_frame,
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
    mark_last_position_prediction: bool = MARK_LAST_PREDICTED_POSITION,
):
    """
    Function to draw information about the detected elements in the image frame.
    <br/><br/>

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    if mark_new_stars and new_star_positions is not None:
        for star in new_star_positions:
            draw_position(show_frame, star)

    if mark_tracked_stars and tracked_stars is not None:
        for star in tracked_stars:
            draw_position(show_frame, star.last_detected_position)

    if mark_shooting_stars and shooting_stars is not None:
        for star in shooting_stars:
            if mark_movement_vector:
                draw_path(show_frame, star)

            draw_position(show_frame, star.last_detected_position, color=(0, 200, 200))

            if mark_next_expected_position:
                draw_position(
                    show_frame,
                    star.next_expected_position,
                    color=(200, 200, 0),
                    thickness=1,
                )
            if mark_last_position_prediction:
                draw_position(
                    show_frame,
                    star.last_predicted_position,
                    color=(200, 0, 200),
                    thickness=1,
                )

    if mark_satellite and satellite is not None:
        if mark_movement_vector:
            draw_path(show_frame, satellite)

        draw_position(show_frame, satellite.last_detected_position, color=(0, 200, 0))

        if mark_next_expected_position:
            draw_position(
                show_frame,
                satellite.next_expected_position,
                color=(200, 200, 0),
                thickness=1,
            )
        if mark_last_position_prediction:
            draw_position(
                show_frame,
                satellite.last_predicted_position,
                color=(200, 0, 200),
                thickness=1,
            )


def draw_position(
    show_frame,
    target: tuple[int, int],
    radius: int = PX_SENSITIVITY,
    color: tuple = None,
    colorized_tracked_stars: bool = COLORIZED_TRACKED_STARS,
    thickness: int = 2,
):
    """
    Function to draw in the given frame a circle around the given position.
    <br/><br/>

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    <br/><br/>

    Note: colorized_tracked_stars parameter is ignored if color parameter is not None.
    """

    if color is None:
        if colorized_tracked_stars and isinstance(target, Star):
            draw_color = target.color
        else:
            draw_color = (0, 0, 100)
    else:
        draw_color = color

    cv.circle(
        show_frame,
        center=(
            int(target[0]),
            int(target[1]),
        ),
        radius=radius,
        color=draw_color,
        thickness=thickness,
    )


def draw_path(
    show_frame,
    targets: Star | set[Star],
    color: tuple = (200, 200, 0),
    thickness: int = 1,
):
    """
    Function to draw in the given frame a line through the last detected positions of
    the given objects.
    <br/><br/>

    Note: This function modifies data from 'show_frame' parameter without an explicit
    return statement for memory usage reduction purposes.
    """

    for target in {targets} if isinstance(targets, Star) else targets:
        last_positions = [
            [round(axis) for axis in pos]
            for pos in target.last_positions
            if pos is not None
        ]
        for pos_1, pos_2 in pairwise(last_positions):
            cv.line(
                show_frame,
                pt1=pos_1,
                pt2=pos_2,
                color=color,
                thickness=thickness,
            )


def tracking_phase_video_simulation(satellite, show_frame, frame_center):
    """Function to simulate the tracking phase of the satellite by moving each frame to
    set the target satellite in the center of the video."""

    translation_vector = [
        frame_center[0] - satellite.next_expected_position[0],
        frame_center[1] - satellite.next_expected_position[1],
    ]

    return translate(show_frame, translation_vector[0], translation_vector[1])


def create_export_video_file(vid_cap, output_video_path: str):
    """Function to create a video file to export the processed frames."""

    width = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid_cap.get(cv.CAP_PROP_FPS))
    frame_count = int(vid_cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo format: {width}x{height} px - {fps} fps")
    print("Number of frames:", frame_count)

    output_video = cv.VideoWriter(
        output_video_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not output_video.isOpened():
        sys.exit("\nError: Unable to create video file.")

    return output_video


def export_satellite_log(
    satellite_log: list[Star],
    path_output_sat_log: str = str(PATH_OUTPUT_SAT_LOG),
):
    """Function to export the satellite log into a file."""

    with open(path_output_sat_log, "w", encoding="utf-8-sig") as file:
        print(
            "id;",
            "last_times_detected;",
            "lifetime;",
            "left_lifetime;",
            "detection_confidence;",
            "blinking_freq;",
            "movement_vector;",
            "frames_since_last_detection;",
            "last_detected_position;",
            "next_expected_position;",
            "last_predicted_position",
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
                f"{star.frames_since_last_detection};",
                f"{star.last_detected_position};",
                f"{star.next_expected_position};",
                f"{star.last_predicted_position};",
                f"{star.last_positions};",
                file=file,
            )


if __name__ == "__main__":
    main()
