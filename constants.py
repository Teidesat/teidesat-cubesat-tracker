#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to define every needed constant for the program.
"""

from datetime import datetime
from pathlib import Path

# * Constants
STAR_DETECTOR_THRESHOLD = 50
FAST = True
MIN_PRUNE_DISTANCE = 20.0

SAT_DESIRED_BLINKING_FREQ = 15.0
MOVEMENT_THRESHOLD = 3.0
PX_SENSITIVITY = 8

PATH_FRAME = Path("./data/images/original.jpg")
PATH_VIDEO = Path("./data/videos/video4.mp4")
PATH_SAT_LOG = Path("./data/logs/satellite_log.csv")

RGB_IMAGE = True
VIDEO_FROM_CAMERA = False
VIDEO_FPS = 60.0

SHOW_VIDEO_RESULT = True
COLORIZED_TRACKED_STARS = False

OUTPUT_RAW_VIDEO_TO_FILE = False
PATH_OUTPUT_RAW_VIDEO = Path(
    "./data/videos/video_output_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "_raw.mp4"
)

OUTPUT_PROCESSED_VIDEO_TO_FILE = True
PATH_OUTPUT_PROCESSED_VIDEO = Path(
    "./data/videos/video_output_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "_processed.mp4"
)

DEFAULT_LEFT_LIFETIME = 10
DEFAULT_VECTOR = (0.0, 0.0)

MIN_DETECTION_CONFIDENCE = 20

MIN_HISTORY_LENGTH = 10
MAX_HISTORY_LENGTH = 20

REMOVE_OUTLIERS = True
MAX_OUTLIER_THRESHOLD = 1.5
MAX_MOVE_DISTANCE = 10.0

FREQUENCY_THRESHOLD = 3.0
