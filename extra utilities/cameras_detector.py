#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to detect all available cameras.
"""

import cv2 as cv


def main():
    available_ports, working_ports = detect_cameras()

    print("Ignore index warnings/errors if any.", end="\n\n")
    print("Found cameras:")

    for working_port, width, height, fps in working_ports:
        print(
            f"  * Port {working_port} is working and reads images "
            + f"({width}x{height} - {fps} fps)"
        )

    for available_port, width, height, fps in available_ports:
        print(
            f"  * Port {available_port} for camera ({width}x{height} - {fps} fps) "
            + "is present but does not reads."
        )

    if not available_ports and not working_ports:
        print("  * No cameras found.")


def detect_cameras():
    """Function to detect all available cameras."""

    working_ports = []
    available_ports = []

    for dev_port in range(100):
        vid_cap = cv.VideoCapture(dev_port)

        if vid_cap.isOpened():
            is_reading, _ = vid_cap.read()

            width = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid_cap.get(cv.CAP_PROP_FPS))

            if is_reading:
                working_ports.append((dev_port, width, height, fps))
            else:
                available_ports.append((dev_port, width, height, fps))

    return available_ports, working_ports


if __name__ == "__main__":
    main()
