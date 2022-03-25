#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module docstring """  # ToDo: redact docstring

import sys

from pathlib import Path

import cv2 as cv

INPUT = Path("../data/videos/video1.mp4")
OUTPUT = Path("../data/frames")


def store_frames(video_path: Path, output_dir: Path):
    """ Store frames from a video to process """

    output_subdir = output_dir / video_path.stem
    output_subdir.mkdir(parents=True, exist_ok=True)

    vidcap = cv.VideoCapture(str(video_path))
    if not vidcap.isOpened():
        sys.exit("Error: Unable to open camera.")

    count = 0
    success, image = vidcap.read()
    while success:
        cv.imwrite(f"{output_subdir}/frame{count}.jpg", image)
        print("Saved frame", count)
        success, image = vidcap.read()
        count += 1


if __name__ == "__main__":
    store_frames(INPUT, OUTPUT)
