#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to define every needed constant for the program.
"""

from datetime import datetime
from pathlib import Path


##################
# Input settings #
##################

# Type of the input stream to use, the supported types are:
#   - "VIDEO_FILE" for pre-recorded videos.
#   - "WEBCAM" for USB webcams detected by OpenCV.
#   - "ZWOASI" for ZWO ASI cameras detected by the ZWO ASI SDK.
SOURCE_TYPE = "VIDEO_FILE"

# If using a camera as input source, index of the camera to use.
#  The default camera value is 0.
CAMERA_INDEX = 0

# If using a video file as input source, path to the video file to use.
PATH_INPUT_VIDEO = Path("./data/videos/video4.mp4")

# Default frames per second of the input stream (only used if it couldn't be retrieved
#  from the input source's information).
VIDEO_FPS = 60.0


###################
# Output settings #
###################

# If true, the program will show the processed video in a window.
SHOW_VIDEO_RESULT = True

# If true, the detected satellite will be centered in the video.
SIMULATE_TRACKING = True

# If true, the program will draw a circle around the detected stars.
MARK_DETECTED_STARS = False
# If true, the program will draw a circle around the tracked stars.
MARK_TRACKED_STARS = True
# If true, the program will draw a circle around the shooting stars.
MARK_SHOOTING_STARS = True
# If true, the program will draw a circle around the detected satellite.
MARK_SATELLITE = True
# If true, the program will draw a line representing the movement vector of the
#  shooting stars and the satellite.
MARK_MOVEMENT_VECTOR = True
# If true, the program will draw a circle around the next expected position of the
#  shooting stars and the satellite.
MARK_NEXT_EXPECTED_POSITION = True
# If true, the program will draw a circle around the last predicted position of the
#  shooting stars and the satellite.
MARK_LAST_PREDICTED_POSITION = True

# If true, the program will draw the tracked stars in different colors.
COLORIZED_TRACKED_STARS = False

# Default color of the circle to draw around each star.
MARK_POSITION_COLOR = (0, 0, 100)
# Default color of the line to draw the movement vector of each star.
MARK_MOVEMENT_VECTOR_COLOR = (200, 200, 0)
# Radius of the circle to draw around each star.
MARK_RADIUS = 8
# Thickness of the line to draw around each star.
MARK_THICKNESS = 2

# If true, the program will dump the input stream frames into a video file.
OUTPUT_RAW_VIDEO_TO_FILE = True
# Path to the file in which the input stream frames will be dumped.
PATH_OUTPUT_RAW_VIDEO = Path(
    "./data/videos/video_output_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "_raw.mp4",
)

# If true, the program will dump the processed frames into a video file.
OUTPUT_PROCESSED_VIDEO_TO_FILE = True
# Path to the file in which the processed frames will be dumped.
PATH_OUTPUT_PROCESSED_VIDEO = Path(
    "./data/videos/video_output_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + "_processed.mp4",
)


# If true, the program will log the detected satellite data into a file.
OUTPUT_SAT_LOG_TO_FILE = True
# Path to the file in which the satellite data will be logged.
PATH_OUTPUT_SAT_LOG = Path(
    "./data/logs/satellite_log_"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    + ".csv",
)


#################
# Star settings #
#################

# Minimum brightness to be considered a star.
STAR_DETECTOR_THRESHOLD = 50

# If true, the program will prune close points to avoid duplicate stars
#  else, it will keep all the detected stars.
# Note: If enabled, the performance of the program could be highly reduced.
PRUNE_CLOSE_POINTS = False
# Minimum distance between two points to be considered different stars.
MIN_PRUNE_DISTANCE = 20.0

# Number of frames a star will be stored without being detected.
DEFAULT_LEFT_LIFETIME = 10
# Maximum number of frames to store as the history of a star.
MAX_HISTORY_LENGTH = 20


################################
# Satellite detection settings #
################################

# Desired blinking frequency of the satellite on detection phase.
SAT_DESIRED_BLINKING_FREQ = 15.0
# Minimum blinking difference to consider a star as the satellite.
FREQUENCY_THRESHOLD = 3.0
# Minimum confidence to consider a star as the satellite based on its blinking
#  frequency and relative movement velocity.
MIN_DETECTION_CONFIDENCE = 20


########################################
# Movement vector computation settings #
########################################

# Default movement vector of a star.
DEFAULT_VECTOR = (0.0, 0.0)
# Minimum distance to consider that a star has moved.
MOVEMENT_THRESHOLD = 3.0
# Minimum number of frames to compute the movement vector of a star.
MIN_HISTORY_LENGTH = 10

# If true, the program will remove outliers when computing the average movement
#  vector of a star.
REMOVE_OUTLIERS = True
# Threshold to consider a movement vector as an outlier.
MAX_OUTLIER_THRESHOLD = 1.5
# Maximum distance difference to consider a movement vector as an outlier.
MAX_MOVE_DISTANCE = 10.0
# Method to compute the average movement vector of a star.
MOVEMENT_VECTOR_COMPUTATION_METHOD = "mean"
