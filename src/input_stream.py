#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with the implementation of the camera class.
"""

from os import getenv as os__get_env_var
from pathlib import Path
from sys import exit as sys__exit

import cv2 as cv
import zwoasi

from constants import (
    SOURCE_TYPE,
    CAMERA_INDEX,
    PATH_INPUT_VIDEO,
    VIDEO_FPS,  # ToDo: Try to get from input source
)


class InputStream:
    def __init__(
        self,
        source_type: str = SOURCE_TYPE,
        camera_index: int = CAMERA_INDEX,
        path_input_video: Path = PATH_INPUT_VIDEO,
    ):
        self.source_type = source_type
        self.camera_index = camera_index
        self.path_input_video = path_input_video

        self._input_stream = None

        if source_type == "ZWOASI":
            self.__init__zwo_asi_camera__()

        elif source_type == "WEBCAM":
            print(f"Initializing webcam #{self.camera_index} using OpenCV")
            self.__init__opencv_video_capture__(self.camera_index)

        elif source_type == "VIDEO_FILE":
            print(f"Processing video from: {self.path_input_video} using OpenCV")
            self.__init__opencv_video_capture__(str(self.path_input_video))

        else:
            sys__exit("\nError: Invalid source type.")

    def __init__zwo_asi_camera__(self):
        zwo_asi_lib = os__get_env_var("ZWO_ASI_LIB")
        zwoasi.init(zwo_asi_lib)

        if not zwoasi.get_num_cameras():
            print("No cameras found")
            exit(0)

        print(f"Initializing ZWO ASI camera #{self.camera_index}")
        self._input_stream = zwoasi.Camera(self.camera_index)

        # Get camera information
        self.zwo_camera_info = self._input_stream.get_camera_property()

        print("Camera info:")
        for k, v in self.zwo_camera_info.items():
            print(f"  {k}: {v}")

        self.is_color_camera = self.zwo_camera_info["IsColorCam"]
        self.fps = VIDEO_FPS  # ToDo: value not available, may have to be calculated

        # Get frame dimensions and calculate its center
        self.frame_width = self.zwo_camera_info["MaxWidth"]
        self.frame_height = self.zwo_camera_info["MaxHeight"]
        self.frame_center = (self.frame_width / 2, self.frame_height / 2)

        # Use minimum USB bandwidth permitted
        self._input_stream.set_control_value(
            zwoasi.ASI_BANDWIDTHOVERLOAD,
            self._input_stream.get_controls()["BandWidth"]["MinValue"],
        )

        # Set some sensible defaults. They will need adjusting depending upon
        # the sensitivity, lens and lighting conditions used.
        self._input_stream.disable_dark_subtract()

        # ToDo: extract setting values to parameters and enable modifications on the fly
        self._input_stream.set_control_value(zwoasi.ASI_GAIN, 200)
        self._input_stream.set_control_value(zwoasi.ASI_EXPOSURE, 500)
        self._input_stream.set_control_value(zwoasi.ASI_WB_B, 99)
        self._input_stream.set_control_value(zwoasi.ASI_WB_R, 75)
        self._input_stream.set_control_value(zwoasi.ASI_GAMMA, 50)
        self._input_stream.set_control_value(zwoasi.ASI_BRIGHTNESS, 50)
        self._input_stream.set_control_value(zwoasi.ASI_FLIP, 0)

        print("Enabling video mode")
        self._input_stream.start_video_capture()

        # Set the timeout, units are ms
        timeout = (
            self._input_stream.get_control_value(zwoasi.ASI_EXPOSURE)[0] / 1000
        ) * 2 + 500  # ToDo: Decript this values and adjust them for better FPS
        self._input_stream.default_timeout = timeout

    def __init__opencv_video_capture__(self, ocv_video_source):
        self._input_stream = cv.VideoCapture(ocv_video_source)

        if not self._input_stream.isOpened():
            sys__exit("\nError: Unable to open video source.")

        # get if camera captures color or monochrome based on the number of channels of
        #  the first frame; if 3, it is color; if 1, it is monochrome
        self.is_color_camera = self._input_stream.read()[1].shape[2] == 3

        self.fps = int(self._input_stream.get(cv.CAP_PROP_FPS))

        self.frame_width = int(self._input_stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._input_stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_center = (self.frame_width / 2, self.frame_height / 2)

    def get_next_frame(self):
        if self.source_type == "ZWOASI":
            try:
                return self._input_stream.capture_video_frame()
            except zwoasi.ZWO_IOError:
                return None

        elif self.source_type == "WEBCAM" or self.source_type == "VIDEO_FILE":
            success, next_frame = self._input_stream.read()
            return next_frame if success else None

        else:
            sys__exit("\nError: Invalid source type.")

    def release(self):
        if self.source_type == "ZWOASI":
            self._input_stream.stop_video_capture()
            self._input_stream.close()

        elif self.source_type == "WEBCAM" or self.source_type == "VIDEO_FILE":
            self._input_stream.release()

        else:
            sys__exit("\nError: Invalid source type.")
