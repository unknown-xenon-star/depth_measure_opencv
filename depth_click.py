# depth_click.py

import cv2
import numpy as np

from depth_map import (
    compute_depth,
    run_static,
    run_live
)

from input import (
    MODE,
    LEFT_IMAGE,
    RIGHT_IMAGE,
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX
)


# =========================
# GLOBALS
# =========================

current_left_frame = None
current_depth_map = None


# =========================
# MOUSE CALLBACK
# =========================

def mouse_callback(event, x, y, flags, param):

    global current_left_frame
    global current_depth_map

    if event == cv2.EVENT_LBUTTONDOWN:

        if current_depth_map is None:
            return

        depth_value = current_depth_map[y, x]

        print(f"Depth at ({x}, {y}) : {depth_value:.2f} m")

        display_frame = current_left_frame.copy()

        text = f"Depth at ({x}, {y}) : {depth_value:.2f} m"

        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(
            display_frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Left Image", display_frame)


# =========================
# STATIC MODE
# =========================

def run_static_click():

    global current_left_frame
    global current_depth_map

    left = cv2.imread(LEFT_IMAGE)
    right = cv2.imread(RIGHT_IMAGE)

    if left is None or right is None:
        raise FileNotFoundError("Could not load images.")

    disparity_colormap, depth_map = compute_depth(left, right)

    current_left_frame = left
    current_depth_map = depth_map

    cv2.namedWindow("Left Image")
    cv2.setMouseCallback("Left Image", mouse_callback)

    while True:

        cv2.imshow("Left Image", current_left_frame)
        cv2.imshow("Disparity Map", disparity_colormap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# =========================
# LIVE MODE
# =========================

def run_live_click():

    global current_left_frame
    global current_depth_map

    cap_left = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    cap_right = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if not cap_left.isOpened():
        raise RuntimeError("Could not open left camera")

    if not cap_right.isOpened():
        raise RuntimeError("Could not open right camera")

    cv2.namedWindow("Left Image")
    cv2.setMouseCallback("Left Image", mouse_callback)

    while True:

        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Frame capture failed")
            break

        disparity_colormap, depth_map = compute_depth(
            frame_l,
            frame_r
        )

        current_left_frame = frame_l
        current_depth_map = depth_map

        cv2.imshow("Left Image", current_left_frame)
        cv2.imshow("Disparity Map", disparity_colormap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if MODE == "static":
        run_static_click()

    elif MODE == "live":
        run_live_click()

    else:
        print("Invalid MODE")