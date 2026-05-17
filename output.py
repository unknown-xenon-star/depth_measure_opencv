# stereo_viewer.py

import cv2
import numpy as np

from input import (
    MODE,
    LEFT_IMAGE,
    RIGHT_IMAGE,
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX
)

from depth_map import compute_depth


# =========================
# GLOBALS
# =========================

clicked_x = None
clicked_y = None

current_depth_map = None
current_left_frame = None
current_right_frame = None
current_disparity_map = None


# =========================
# DRAW CROSSHAIR
# =========================

def draw_crosshair(img, x, y, color=(0, 255, 0)):

    size = 15

    cv2.line(img, (x - size, y), (x + size, y), color, 2)
    cv2.line(img, (x, y - size), (x, y + size), color, 2)

    cv2.circle(img, (x, y), 4, color, -1)


# =========================
# MOUSE CALLBACK
# =========================

def mouse_callback(event, x, y, flags, param):

    global clicked_x
    global clicked_y

    global current_depth_map

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_x = x
        clicked_y = y

        depth_value = current_depth_map[y, x]

        print(f"Depth at ({x}, {y}) : {depth_value:.2f} m")


# =========================
# BUILD DISPLAY
# =========================

def build_display():

    left_display = current_left_frame.copy()
    right_display = current_right_frame.copy()
    depth_display = current_disparity_map.copy()

    if clicked_x is not None and clicked_y is not None:

        depth_value = current_depth_map[clicked_y, clicked_x]

        label = (
            f"Depth: {depth_value:.2f} m"
        )

        draw_crosshair(
            left_display,
            clicked_x,
            clicked_y
        )

        draw_crosshair(
            right_display,
            clicked_x,
            clicked_y
        )

        draw_crosshair(
            depth_display,
            clicked_x,
            clicked_y
        )

        cv2.putText(
            left_display,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    combined = np.hstack([
        left_display,
        right_display,
        depth_display
    ])

    return combined


# =========================
# STATIC MODE
# =========================

def run_static():

    global current_left_frame
    global current_right_frame
    global current_depth_map
    global current_disparity_map

    left = cv2.imread(LEFT_IMAGE)
    right = cv2.imread(RIGHT_IMAGE)

    if left is None or right is None:
        raise FileNotFoundError("Could not load images.")

    disparity_map, depth_map = compute_depth(
        left,
        right
    )

    current_left_frame = left
    current_right_frame = right
    current_depth_map = depth_map
    current_disparity_map = disparity_map

    cv2.namedWindow("Stereo Depth Viewer")
    cv2.setMouseCallback(
        "Stereo Depth Viewer",
        mouse_callback
    )

    while True:

        display = build_display()

        cv2.imshow(
            "Stereo Depth Viewer",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# =========================
# LIVE MODE
# =========================

def run_live():

    global current_left_frame
    global current_right_frame
    global current_depth_map
    global current_disparity_map

    cap_left = cv2.VideoCapture(
        LEFT_CAMERA_INDEX
    )

    cap_right = cv2.VideoCapture(
        RIGHT_CAMERA_INDEX
    )

    if not cap_left.isOpened():
        raise RuntimeError(
            "Could not open left camera"
        )

    if not cap_right.isOpened():
        raise RuntimeError(
            "Could not open right camera"
        )

    cv2.namedWindow("Stereo Depth Viewer")

    cv2.setMouseCallback(
        "Stereo Depth Viewer",
        mouse_callback
    )

    while True:

        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Frame capture failed")
            break

        disparity_map, depth_map = compute_depth(
            frame_l,
            frame_r
        )

        current_left_frame = frame_l
        current_right_frame = frame_r
        current_depth_map = depth_map
        current_disparity_map = disparity_map

        display = build_display()

        cv2.imshow(
            "Stereo Depth Viewer",
            display
        )

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
        run_static()

    elif MODE == "live":
        run_live()

    else:
        print("Invalid MODE")