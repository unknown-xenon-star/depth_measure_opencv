from time import sleep, time
import cv2
import numpy as np
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from collections import deque
from tools.kalman_filter import KalmanDepthTracker
from tools.hsv import add_HSV_filter
from tools.detection import find_object
from tools.disparity_map import disparity_n_depth_map
from tools.extras import masked_percentile_depth



# =============================================================================
# INTRINSIC CALIBRATION VALUES
# =============================================================================

K_LEFT = np.array([
    [734.78419479, 0.0, 335.558714],
    [0.0, 734.67820344, 234.05811661],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_LEFT = np.array([
    [-4.70065931e-01,
      4.76997998e-01,
      1.45712481e-04,
     -2.64300524e-03,
     -5.80831061e-01]
], dtype=np.float64)

# SAME CAMERA MODEL
K_RIGHT = K_LEFT.copy()
DIST_RIGHT = DIST_LEFT.copy()

# =============================================================================
# EXTRINSIC CALIBRATION VALUES
# =============================================================================
# Example values
# REPLACE THESE WITH YOUR stereoCalibrate() OUTPUT

R = np.array([
    [0.9998, 0.0010, -0.0170],
    [-0.0011, 1.0000, -0.0020],
    [0.0170, 0.0020, 0.9998]
], dtype=np.float64)

T = np.array([
    [-6.5],
    [0.0],
    [0.0]
], dtype=np.float64)

# =============================================================================
# IMAGE SIZE
# =============================================================================

FRAME_SIZE = (640, 480)

# =============================================================================
# STEREO RECTIFICATION
# =============================================================================

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_LEFT,
    DIST_LEFT,
    K_RIGHT,
    DIST_RIGHT,
    FRAME_SIZE,
    R,
    T
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    K_LEFT,
    DIST_LEFT,
    R1,
    P1,
    FRAME_SIZE,
    cv2.CV_16SC2
)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    K_RIGHT,
    DIST_RIGHT,
    R2,
    P2,
    FRAME_SIZE,
    cv2.CV_16SC2
)

# =============================================================================
# DEPTH CONSTANTS
# =============================================================================

BASELINE = 6.5  # cm
FOCAL_LENGTH = P1[0, 0]

DISPARITY_OFFSET = 1.0
DISP_SCALE = 0.75

HSV_MASK = [(0, 96, 78), (80, 255, 220)]

MIN_AREA = 50

MORPH_K = np.ones((5, 5), dtype=np.uint8)

# =============================================================================
# SGBM
# =============================================================================

sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 8,
    blockSize=5,

    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,

    disp12MaxDiff=1,
    uniquenessRatio=10,

    speckleWindowSize=100,
    speckleRange=32,

    mode=cv2.STEREO_SGBM_MODE_SGBM
)

kf = KalmanDepthTracker()

# =============================================================================
# LIVE DEPTH PLOT
# =============================================================================

plt.ion()

fig, ax = plt.subplots(figsize=(8, 4))

depth_history = deque(maxlen=100)

line, = ax.plot([], [])

ax.set_title("Live Depth Plot")
ax.set_xlabel("Frame")
ax.set_ylabel("Depth (cm)")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.show(block=False)

# =============================================================================
# FUNCTIONS
# =============================================================================




# =============================================================================
# CAMERA SETUP
# =============================================================================

cap_left = Picamera2(0)
cap_right = Picamera2(1)

cap_left.configure(
    cap_left.create_preview_configuration(
        main={"size": FRAME_SIZE}
    )
)

cap_right.configure(
    cap_right.create_preview_configuration(
        main={"size": FRAME_SIZE}
    )
)

cap_left.start()
cap_right.start()

sleep(2)

# =============================================================================
# MAIN LOOP
# =============================================================================

frame_count = 0
time_start = time()

while True:

    frame_count += 1

    frame_left = cap_left.capture_array()
    frame_right = cap_right.capture_array()

    frame_left = cv2.cvtColor(
        frame_left,
        cv2.COLOR_RGB2BGR
    )

    frame_right = cv2.cvtColor(
        frame_right,
        cv2.COLOR_RGB2BGR
    )

    # =========================================================================
    # RECTIFICATION
    # =========================================================================

    frame_left = cv2.remap(
        frame_left,
        left_map1,
        left_map2,
        cv2.INTER_LINEAR
    )

    frame_right = cv2.remap(
        frame_right,
        right_map1,
        right_map2,
        cv2.INTER_LINEAR
    )

    # =========================================================================
    # MASKING
    # =========================================================================

    mask_left = add_HSV_filter(frame_left, 7, HSV_MASK)
    mask_right = add_HSV_filter(frame_right, 7, HSV_MASK)

    center_right, bbox_right = find_object(
        frame_right,
        mask_right
    )

    center_left, _ = find_object(
        frame_left,
        mask_left
    )

    tracking = (
        center_left is not None
        and
        center_right is not None
    )

    depth = None

    if tracking:

        disparity_vis, depth_map = disparity_n_depth_map(
            mask_left,
            mask_right
        )

        depth = masked_percentile_depth(
            depth_map,
            mask_right,
            bbox_right
        )

        # depth = kf.update(depth)

        if depth is not None:

            print(f"Depth: {depth:.2f} cm")

            depth_history.append(depth)

            x = np.arange(len(depth_history))

            line.set_xdata(x)
            line.set_ydata(depth_history)

            ax.set_xlim(
                0,
                max(100, len(depth_history))
            )

            current_max = max(depth_history)

            ax.set_ylim(
                0,
                max(100, current_max + 20)
            )

            if frame_count % 5 == 0:

                fig.canvas.draw()
                fig.canvas.flush_events()

    else:

        disparity_vis = np.zeros(
            (480, 640),
            dtype=np.uint8
        )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    if depth is not None:

        cv2.putText(
            frame_right,
            f"Depth: {depth:.2f} cm",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow(
        "LEFT RECTIFIED",
        frame_left
    )

    cv2.imshow(
        "RIGHT RECTIFIED",
        frame_right
    )

    cv2.imshow(
        "DISPARITY",
        disparity_vis
    )

    cv2.imshow(
        "MASK",
        mask_right
    )

    if cv2.pollKey() & 0xFF == ord('q'):

        total_time = time() - time_start

        print("\n======================")
        print(f"TOTAL FRAMES : {frame_count}")
        print(f"TOTAL TIME   : {total_time:.2f}")
        print(f"FPS          : {frame_count/total_time:.2f}")
        print("======================")

        break

# =============================================================================
# CLEANUP
# =============================================================================

cap_left.stop()
cap_right.stop()

cv2.destroyAllWindows()
