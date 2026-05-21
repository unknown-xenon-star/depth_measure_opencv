from time import sleep, time
import cv2
import numpy as np
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# INTRINSIC CALIBRATION VALUES
# =============================================================================

K_LEFT = np.array([
    [734.78419479, 0.0, 335.558714],
    [0.0, 734.67820344, 234.05811661],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_LEFT = np.array([
    [-4.70065931e-01,
      4.76997998e-01,
      1.45712481e-04,
     -2.64300524e-03,
     -5.80831061e-01]
], dtype=np.float32)

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
], dtype=np.float32)

T = np.array([
    [-6.5],
    [0.0],
    [0.0]
], dtype=np.float32)

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

# =============================================================================
# KALMAN FILTER
# =============================================================================

class KalmanDepthFilter:

    def __init__(self):

        self.x = np.array([
            [0.0],
            [0.0]
        ])

        self.P = np.eye(2) * 1000

        dt = 1.0

        self.F = np.array([
            [1, dt],
            [0, 1]
        ])

        self.H = np.array([
            [1, 0]
        ])

        self.R = np.array([
            [10]
        ])

        self.Q = np.array([
            [0.01, 0],
            [0, 0.01]
        ])

        self.initialized = False

    def update(self, measurement):

        self.x = self.F @ self.x

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        if measurement is None:
            return float(self.x[0, 0])

        if not self.initialized:

            self.x[0, 0] = measurement
            self.x[1, 0] = 0

            self.initialized = True

            return measurement

        z = np.array([
            [measurement]
        ])

        y = z - self.H @ self.x

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        K = (
            self.P @ self.H.T
            @ np.linalg.inv(S)
        )

        self.x = self.x + K @ y

        I = np.eye(2)

        self.P = (
            I - K @ self.H
        ) @ self.P

        return float(self.x[0, 0])

kf = KalmanDepthFilter()

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

def hsv_mask(frame):

    blur = cv2.GaussianBlur(frame, (7, 7), 0)

    hsv = cv2.cvtColor(
        blur,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        HSV_MASK[0],
        HSV_MASK[1]
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        MORPH_K,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        MORPH_K,
        iterations=2
    )

    return mask

def find_object(frame, mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < MIN_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(c)

    center = (
        x + w // 2,
        y + h // 2
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        2
    )

    cv2.circle(
        frame,
        center,
        5,
        (0, 0, 0),
        -1
    )

    return center, [[x, y], [x + w, y + h]]

def disparity_n_depth_map(left, right):

    h, w = left.shape[:2]

    if DISP_SCALE != 1.0:

        dsize = (
            int(w * DISP_SCALE),
            int(h * DISP_SCALE)
        )

        left = cv2.resize(left, dsize)
        right = cv2.resize(right, dsize)

    raw = sgbm.compute(
        left,
        right
    ).astype(np.float32) / 16.0

    if DISP_SCALE != 1.0:

        raw = cv2.resize(
            raw,
            (w, h)
        )

        raw /= DISP_SCALE

    valid = raw > 0

    depth = np.zeros_like(raw)

    depth[valid] = (
        FOCAL_LENGTH * BASELINE
    ) / (
        raw[valid] + DISPARITY_OFFSET
    )

    depth[~np.isfinite(depth)] = 0

    disp_vis = cv2.normalize(
        raw,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return disp_vis, depth

def masked_percentile_depth(
    depth_map,
    mask,
    bbox,
    percentile=50
):

    (x1, y1), (x2, y2) = bbox

    values = depth_map[
        y1:y2,
        x1:x2
    ][
        mask[y1:y2, x1:x2] > 0
    ]

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if values.size == 0:
        return None

    return float(
        np.percentile(values, percentile)
    )

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

    mask_left = hsv_mask(frame_left)
    mask_right = hsv_mask(frame_right)

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

        depth = kf.update(depth)

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
