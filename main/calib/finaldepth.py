# ============================================================
# STEREO RECTIFICATION + DEPTH ESTIMATION
# ============================================================

from time import time, sleep
import cv2
import numpy as np
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from collections import deque

# ============================================================
# INTRINSIC PARAMETERS
# ============================================================

K_left = np.array([
    [755.79240334, 0.0, 336.47450076],
    [0.0, 755.36361434, 255.72185014],
    [0.0, 0.0, 1.0]
])

dist_left = np.array([
    [-5.02451202e-01,
      7.33283229e-01,
     -4.02834197e-03,
      8.96470971e-06,
     -1.55985928e+00]
])

K_right = np.array([
    [734.90387831, 0.0, 330.55013381],
    [0.0, 735.03555766, 236.51655862],
    [0.0, 0.0, 1.0]
])

dist_right = np.array([
    [-0.46708432,
      0.5110633,
     -0.00213905,
     -0.00105646,
     -0.80638573]
])

# ============================================================
# EXTRINSIC PARAMETERS
# ============================================================

R = np.array([
    [ 0.99356494,  0.11298987,  0.0078742 ],
    [-0.1126464 ,  0.99300683, -0.03533031],
    [-0.0118111 ,  0.03421596,  0.99934467]
])

T = np.array([
    [ 4.87716352],
    [-0.12075234],
    [-1.89061552]
])

# ============================================================
# CAMERA SETTINGS
# ============================================================

FRAME_SIZE = (640, 480)

BASELINE = 5.23     # cm
FOCAL_LENGTH = 755  # pixels

HSV_MASK = [(0, 96, 78), (80, 255, 220)]

MIN_AREA = 50

MORPH_K = np.ones((5,5), np.uint8)

# ============================================================
# STEREO RECTIFICATION
# ============================================================

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_left,
    dist_left,
    K_right,
    dist_right,
    FRAME_SIZE,
    R,
    T
)

# ============================================================
# RECTIFICATION MAPS
# ============================================================

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    K_left,
    dist_left,
    R1,
    P1,
    FRAME_SIZE,
    cv2.CV_16SC2
)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    K_right,
    dist_right,
    R2,
    P2,
    FRAME_SIZE,
    cv2.CV_16SC2
)

# ============================================================
# SGBM
# ============================================================

stereo = cv2.StereoSGBM_create(
    minDisparity = 0,
    numDisparities = 16 * 8,
    blockSize = 5,

    P1 = 8 * 3 * 5**2,
    P2 = 32 * 3 * 5**2,

    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

# ============================================================
# KALMAN FILTER
# ============================================================

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

        self.P = self.F @ self.P @ self.F.T + self.Q

        if measurement is None:
            return float(self.x[0,0])

        if not self.initialized:

            self.x[0,0] = measurement
            self.initialized = True

            return measurement

        z = np.array([
            [measurement]
        ])

        y = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(2)

        self.P = (I - K @ self.H) @ self.P

        return float(self.x[0,0])

kf = KalmanDepthFilter()

# ============================================================
# LIVE GRAPH
# ============================================================

plt.ion()

fig, ax = plt.subplots(figsize=(8,4))

depth_history = deque(maxlen=100)

line, = ax.plot([], [])

ax.set_title("Live Depth Plot")

ax.set_xlabel("Frame")

ax.set_ylabel("Depth (cm)")

ax.set_ylim(0,100)

ax.set_xlim(0,100)

plt.show(block=False)

# ============================================================
# HSV MASK
# ============================================================

def hsv_mask(frame):

    blur = cv2.GaussianBlur(frame, (7,7), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        HSV_MASK[0],
        HSV_MASK[1]
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        MORPH_K
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        MORPH_K
    )

    return mask

# ============================================================
# OBJECT DETECTION
# ============================================================

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
        (x,y),
        (x+w,y+h),
        (0,255,255),
        2
    )

    cv2.circle(
        frame,
        center,
        5,
        (0,0,255),
        -1
    )

    return center, [[x,y],[x+w,y+h]]

# ============================================================
# DEPTH FUNCTION
# ============================================================

def calculate_depth(disparity):

    if disparity <= 0:
        return None

    depth = (
        FOCAL_LENGTH * BASELINE
    ) / disparity

    return depth

# ============================================================
# CAMERA SETUP
# ============================================================

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

# ============================================================
# MAIN LOOP
# ============================================================

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

    # ========================================================
    # RECTIFICATION
    # ========================================================

    rect_left = cv2.remap(
        frame_left,
        left_map1,
        left_map2,
        cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(
        frame_right,
        right_map1,
        right_map2,
        cv2.INTER_LINEAR
    )

    # ========================================================
    # MASKS
    # ========================================================

    mask_left = hsv_mask(rect_left)

    mask_right = hsv_mask(rect_right)

    # ========================================================
    # OBJECT DETECTION
    # ========================================================

    center_left, bbox_left = find_object(
        rect_left,
        mask_left
    )

    center_right, bbox_right = find_object(
        rect_right,
        mask_right
    )

    tracking = (
        center_left is not None
        and
        center_right is not None
    )

    disparity_vis = np.zeros(
        (480,640),
        dtype=np.uint8
    )

    depth = None

    # ========================================================
    # DEPTH ESTIMATION
    # ========================================================

    if tracking:

        gray_left = cv2.cvtColor(
            rect_left,
            cv2.COLOR_BGR2GRAY
        )

        gray_right = cv2.cvtColor(
            rect_right,
            cv2.COLOR_BGR2GRAY
        )

        disparity = stereo.compute(
            gray_left,
            gray_right
        ).astype(np.float32) / 16.0

        disparity_vis = cv2.normalize(
            disparity,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        disparity_vis = cv2.applyColorMap(
            disparity_vis,
            cv2.COLORMAP_JET
        )

        cx_left = center_left[0]
        cx_right = center_right[0]

        pixel_disparity = abs(
            cx_left - cx_right
        )

        depth = calculate_depth(
            pixel_disparity
        )

        depth = kf.update(depth)

        if depth is not None:

            depth_history.append(depth)

            x = np.arange(
                len(depth_history)
            )

            line.set_xdata(x)

            line.set_ydata(depth_history)

            ax.set_xlim(
                0,
                max(100, len(depth_history))
            )

            ax.set_ylim(
                0,
                max(depth_history) + 20
            )

            if frame_count % 5 == 0:

                fig.canvas.draw()

                fig.canvas.flush_events()

        print(
            f"Depth: {depth:.2f} cm"
            if depth
            else "Depth: NaN"
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    if tracking:

        cv2.putText(
            rect_right,
            f"Depth: {depth:.2f} cm",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    else:

        cv2.putText(
            rect_right,
            "TRACKING LOST",
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    cv2.imshow("LEFT RECTIFIED", rect_left)

    cv2.imshow("RIGHT RECTIFIED", rect_right)

    cv2.imshow("DISPARITY MAP", disparity_vis)

    cv2.imshow("MASK", mask_right)

    key = cv2.waitKey(1)

    if key == ord('q'):

        total_time = time() - time_start

        print("\n================================")
        print("FRAME DATA")
        print("================================")

        print(f"TOTAL FRAMES: {frame_count}")

        print(f"TOTAL TIME: {total_time:.2f}")

        print(f"FPS: {frame_count / total_time:.2f}")

        break

# ============================================================
# CLEANUP
# ============================================================

cap_left.stop()

cap_right.stop()

cv2.destroyAllWindows()