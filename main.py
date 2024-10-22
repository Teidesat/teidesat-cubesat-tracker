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
from itertools import pairwise as itertools__pairwise
from sys import exit as sys__exit
from time import perf_counter as time__perf_counter

import cv2 as cv
from imutils import translate as imutils__translate

from src.image_processor import (
    detect_stars,
    track_stars,
    detect_blinking_star,
    detect_shooting_stars,
)
from src.input_stream import InputStream
from src.star import Star
from constants import (
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
    MARK_RADIUS,
    MARK_THICKNESS,
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

    satellite_detection_test()


def satellite_detection_test(
    sat_desired_blinking_freq: float = SAT_DESIRED_BLINKING_FREQ,
    star_detector_threshold: int = STAR_DETECTOR_THRESHOLD,
    prune_close_points: bool = PRUNE_CLOSE_POINTS,
    min_prune_distance: float = MIN_PRUNE_DISTANCE,
    movement_threshold: float = MOVEMENT_THRESHOLD,
    show_video_result: bool = SHOW_VIDEO_RESULT,
    simulate_tracking: bool = SIMULATE_TRACKING,
    output_sat_log_to_file: bool = OUTPUT_SAT_LOG_TO_FILE,
    path_output_sat_log: str = str(PATH_OUTPUT_SAT_LOG),
    output_raw_video_to_file: bool = OUTPUT_RAW_VIDEO_TO_FILE,
    path_output_raw_video: str = str(PATH_OUTPUT_RAW_VIDEO),
    output_processed_video_to_file: bool = OUTPUT_PROCESSED_VIDEO_TO_FILE,
    path_output_processed_video: str = str(PATH_OUTPUT_PROCESSED_VIDEO),
):
    """Function to detect and track the satellite."""

    # Create input stream object to initialize the source of video frames
    input_stream = InputStream()

    # Create the star detector object
    star_detector = cv.FastFeatureDetector_create(threshold=star_detector_threshold)

    # Initialize the variables to store the information about the tracked stars and the
    #  detected satellite through the video frames
    tracked_stars: set[Star] = set()
    satellite_log: list[Star] = []

    # Initialize the visualization variables
    wait_time = 1  # Default visualization speed
    wait_options = {  # Visualization speed options
        ord("z"): 1,
        ord("x"): 100,
        ord("c"): 1000,
        ord("v"): 0,
    }

    # Initialize the statistics variables
    processed_frames = 0
    start_time = time__perf_counter()

    # Initialize the output video files if needed
    output_raw_video = (
        create_export_video_file(input_stream, path_output_raw_video)
        if output_raw_video_to_file
        else None
    )
    output_processed_video = (
        create_export_video_file(input_stream, path_output_processed_video)
        if output_processed_video_to_file
        else None
    )

    # Initialize the visualization window if needed
    if show_video_result:
        cv.namedWindow("Satellite detection", cv.WINDOW_NORMAL)

    # Start the main processing loop
    while True:

        # Get the next frame from the input stream
        frame = input_stream.get_next_frame()
        if frame is None:
            if input_stream.source_type == "VIDEO_FILE":
                break  # Exiting because the video file has probably ended
            else:
                continue

        processed_frames += 1

        # Convert the frame to grayscale if not already
        grayscale_frame = (
            cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            if input_stream.is_color_camera
            else frame
        )

        # Detect the sky objects in the current frame
        new_star_positions = detect_stars(
            grayscale_frame,
            star_detector,
            prune_close_points,
            min_prune_distance,
        )

        # Update the tracked sky objects information with the new detected positions
        track_stars(
            new_star_positions,
            tracked_stars,
            sat_desired_blinking_freq,
            input_stream.fps,
        )

        # Filter the fast moving objects
        shooting_stars = detect_shooting_stars(tracked_stars, movement_threshold)

        # Get the object with the blinking frequency closest to the desired one
        satellite = detect_blinking_star(shooting_stars)

        # Prepare the frame to show the detected objects
        show_frame = (
            frame.copy()
            if input_stream.is_color_camera
            else cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
        )

        # Draw the detected objects in the frame
        draw_in_frame(
            show_frame,
            new_star_positions,
            tracked_stars,
            shooting_stars,
            satellite,
        )

        if satellite is not None:
            # Save the current satellite information to store as a log
            satellite_log.append(deepcopy(satellite))

            # Transform the frame to simulate the tracking phase if needed
            if simulate_tracking:
                show_frame = tracking_phase_video_simulation(
                    satellite, show_frame, input_stream.frame_center
                )

        # Export the raw frame to the output video file if needed
        if output_raw_video is not None:
            output_raw_video.write(
                frame.copy()
                if input_stream.is_color_camera
                else cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
            )

        # Export the processed frame to the output video file if needed
        if output_processed_video is not None:
            output_processed_video.write(show_frame)

        # Show the processed frame in the visualization window if needed
        if show_video_result:
            cv.imshow("Satellite detection", show_frame)

            # Wait for the user input to change the visualization speed or exit the
            #  program
            key = cv.waitKey(wait_time)

            # Exit the program if the user pressed the 'q' key
            if key == ord("q"):
                print("Exit requested by user.")
                break

            # Update the visualization speed if the user pressed one of the speed keys
            wait_time = wait_options.get(key, wait_time)

    # Print the processing time statistics
    print_time_statistics(processed_frames, start_time)

    # Export the satellite log to a file if needed
    if output_sat_log_to_file:
        export_satellite_log(satellite_log, path_output_sat_log)

    # Notify the user about the created output video files
    if output_raw_video is not None:
        print(f"Raw video saved on '{path_output_raw_video}'")
    if output_processed_video is not None:
        print(f"Processed video saved on '{path_output_processed_video}'")

    # Release the resources and close the visualization window
    input_stream.release()
    cv.destroyAllWindows()


def print_time_statistics(
    processed_frames: int,
    start_time: float,
):
    """Function to print processing time statistics."""

    processing_time = time__perf_counter() - start_time

    print("Processing time statistics:")
    print("  Processed frames:", processed_frames)
    print("  Time needed:", processing_time)
    print("  Estimated FPS:", processed_frames / processing_time)


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
    colorized_tracked_stars: bool = COLORIZED_TRACKED_STARS,
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
            if colorized_tracked_stars and isinstance(star, Star):
                draw_color = star.color
            else:
                draw_color = None

            draw_position(show_frame, star.last_detected_position, color=draw_color)

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
    radius: int = MARK_RADIUS,
    color: tuple = None,
    thickness: int = MARK_THICKNESS,
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
        for pos_1, pos_2 in itertools__pairwise(last_positions):
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

    return imutils__translate(show_frame, translation_vector[0], translation_vector[1])


def create_export_video_file(input_stream, output_video_path: str):
    """Function to create a video file to export the processed frames."""

    output_video = cv.VideoWriter(
        output_video_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        input_stream.fps,
        (input_stream.frame_width, input_stream.frame_height),
    )

    if not output_video.isOpened():
        sys__exit("\nError: Unable to create video file.")

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
