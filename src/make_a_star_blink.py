#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to store all the frames of a video as separate images.
"""

import sys

from pathlib import Path

import cv2 as cv
import numpy as np

INPUT = Path("../data/videos/video1.mp4")
OUTPUT = Path("../data/videos/video_test.mp4")

PX_SENSITIVITY = 10

click_location = ()
remove_star = False


def get_click_location(event, x_coord, y_coord, flags, param):
    """ Mouse callback function. """

    global click_location, remove_star

    if event == cv.EVENT_LBUTTONDOWN:
        click_location = (x_coord, y_coord)
        remove_star = True


def get_avg_color(frame, radius):
    """ Function to get the average surrounding color of a pixel. """

    global click_location

    x_coord, y_coord = click_location
    roi = frame[(y_coord - radius):(y_coord + radius),
                (x_coord - radius):(x_coord + radius)]

    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return avg_color


def make_a_star_blink(input_video_path: Path, output_video_path: Path):
    """ Function to make a star blink by removing it from some frames. """

    global click_location, remove_star

    vidcap = cv.VideoCapture(str(input_video_path))
    if not vidcap.isOpened():
        sys.exit("Error: Unable to open camera.")

    width = vidcap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = vidcap.get(cv.CAP_PROP_FPS)
    frame_count = vidcap.get(cv.CAP_PROP_FRAME_COUNT)
    print(width, height, fps, frame_count)

    desired_blinking_freq = fps / 2
    frames_to_skip = fps / desired_blinking_freq

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(str(output_video_path), fourcc, fps,
                           (int(width), int(height)))

    cv.namedWindow("frame")
    cv.setMouseCallback("frame", get_click_location)

    processed_frames = 0
    success, frame = vidcap.read()
    while success:
        if processed_frames % frames_to_skip == 0:
            video.write(frame)

            success, frame = vidcap.read()
            processed_frames += 1
            print(processed_frames)

        else:
            cv.imshow("frame", frame)

            if remove_star:
                remove_star = False

                color = get_avg_color(frame, PX_SENSITIVITY)
                cv.circle(
                    frame,
                    center=click_location,
                    radius=PX_SENSITIVITY,
                    color=color,
                    thickness=cv.FILLED,
                )
                video.write(frame)

                success, frame = vidcap.read()
                processed_frames += 1
                print(processed_frames)

        if cv.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    make_a_star_blink(INPUT, OUTPUT)
