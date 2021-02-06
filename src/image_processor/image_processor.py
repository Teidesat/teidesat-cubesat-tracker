import cv2
import numpy as np

KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])


def prune_close_points(indices, distance):
    pruned = []
    for i, (y, x) in enumerate(indices):
        duplicated = False
        for yy, xx in indices[i + 1:]:
            d = abs(y - yy) + abs(x - xx)
            if d < 10:
                duplicated = True
                break
        if not duplicated:
            pruned.append((y, x))
    return pruned


def find_stars(image, threshold=20, px_sensitivity=10, fast=True, distance=20):
    edges_x = cv2.filter2D(image, cv2.CV_8U, KERNEL_X)
    edges_y = cv2.filter2D(image, cv2.CV_8U, KERNEL_Y)
    mask = np.minimum(edges_x, edges_y)
    indices = np.argwhere(mask > threshold)

    if fast:
        # A) May give two (or more?) detections for one star
        # B) Not smooth transitions between frames
        indices = np.round(indices / px_sensitivity) * px_sensitivity  # Group stars that are closer
        indices = {(x, y) for y, x in indices}
        return indices
    else:
        indices_round = np.round(indices / px_sensitivity) * px_sensitivity
        # Same as before but now store the original index to recover later the original coordinates
        # Then prune the positions that are too close
        indices_round = {(y, x): i for i, (y, x) in enumerate(indices_round)}
        indices = [(indices[i][1], indices[i][0]) for _, i in indices_round.items()]
        return prune_close_points(indices, distance)
