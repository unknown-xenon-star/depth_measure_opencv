# depth_map.py

import cv2
import numpy as np

from input import (
    MODE,
    LEFT_IMAGE,
    RIGHT_IMAGE,
    LEFT_CAMERA_INDEX,
    RIGHT_CAMERA_INDEX
)


# =========================
# CONFIG
# =========================

FOCAL_LENGTH = 700      # pixels
BASELINE = 0.06         # meters


# =========================
# STEREO SGBM
# =========================

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)


# =========================
# DEPTH COMPUTATION
# =========================

def compute_depth(left_frame, right_frame):

    # Convert to grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)

    # OpenCV scales disparity by 16
    disparity = disparity / 16.0

    # Avoid division by zero
    disparity[disparity <= 0] = 0.1

    # Depth formula
    depth = (FOCAL_LENGTH * BASELINE) / disparity

    # Normalize disparity for display
    disparity_display = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    disparity_display = np.uint8(disparity_display)

    # Apply color map
    disparity_colormap = cv2.applyColorMap(
        disparity_display,
        cv2.COLORMAP_JET
    )

    return disparity_colormap, depth


# =========================
# STATIC MODE
# =========================

def run_static():

    left = cv2.imread(LEFT_IMAGE)
    right = cv2.imread(RIGHT_IMAGE)

    if left is None or right is None:
        raise FileNotFoundError("Could not load images.")

    disparity_map, depth = compute_depth(left, right)

    cv2.imshow("Left Image", cv2.resize(left, (0, 0), fx=0.35, fy=0.35))
    cv2.imshow("Right Image", cv2.resize(right, (0, 0), fx=0.35, fy=0.35))
    cv2.imshow("Disparity / Depth Map", cv2.resize(disparity_map, (0, 0), fx=0.35, fy=0.35))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# LIVE MODE
# =========================

def run_live():

    cap_left = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    cap_right = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if not cap_left.isOpened():
        raise RuntimeError("Could not open left camera")

    if not cap_right.isOpened():
        raise RuntimeError("Could not open right camera")

    while True:

        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Frame capture failed")
            break

        disparity_map, depth = compute_depth(frame_l, frame_r)

        cv2.imshow("Left Camera", frame_l)
        cv2.imshow("Right Camera", frame_r)
        cv2.imshow("Disparity / Depth Map", disparity_map)

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