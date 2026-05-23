from time import time, sleep
import cv2
import numpy as np
from picamera2 import Picamera2 
import matplotlib.pyplot as plt
from collections import deque
from tools.kalman_filter import KalmanDepthTracker
from tools.disparity_map import disparity_n_depth_map
from tools.hsv import add_HSV_filter
from tools.extras import masked_percentile_depth, annotate
from tools.detection import find_object
from config import (
    MASK_HSV
)

# ── Stereo / depth constants ───────────────────────────────────────────────────
BASELINE         = 6.5    # camera baseline [cm]
FOV_DEG          = 82       # horizontal FOV [degrees]
FOCAL_LENGTH     = 220.0    # pixels — tune to your camera
DISPARITY_OFFSET = 1.0      # avoids divide-by-zero at zero disparity
DISP_SCALE       = 0.75      # resize factor fed into SGBM (0.5 = quarter pixels)
HSV_MASK = [(0, 96, 78), (80, 255, 220)]

# ── Detection constants ────────────────────────────────────────────────────────
MIN_AREA = 50
MORPH_K  = np.ones((5, 5), np.uint8)

# ── SGBM — created once, reused every frame ────────────────────────────────────
# numDisparities must be divisible by 16; keep it small for speed.
# blockSize odd, 3–11 is a good range.
_sgbm = cv2.StereoSGBM_create(
    minDisparity      = 0,
    numDisparities    = 16 * 8,     # 192 — must be divisible by 16
    blockSize         = 5,
    P1                = 8  * 3 * 5 ** 2,
    P2                = 32 * 3 * 5 ** 2,
    disp12MaxDiff     = 1,
    uniquenessRatio   = 10,
    speckleWindowSize = 100,
    speckleRange      = 32,
    mode              = cv2.STEREO_SGBM_MODE_SGBM,   # faster than default MODE_HH
)

# ── Core functions ─────────────────────────────────────────────────────────────



kf = KalmanDepthTracker()

# ── Live matplotlib plot ────────────────────────────────────────────────────
plt.ion()

fig, ax = plt.subplots(figsize=(8, 4))

depth_history = deque(maxlen=100)

line, = ax.plot([], [])

ax.set_title("Live Depth Plot")
ax.set_xlabel("Frame")
ax.set_ylabel("Depth (cm)")

ax.set_ylim(0, 100)
ax.set_xlim(0, 100)

plt.show(block=False)



# ── Main loop ──────────────────────────────────────────────────────────────────

# cap = cv2.VideoCapture("assest/video.mp4")
# cap_left = cv2.VideoCapture(0)
# cap_right = cv2.VideoCapture(1)
cap_left = Picamera2(0)
cap_right = Picamera2(1)
cap_left.configure(cap_left.create_preview_configuration(main={"size": (640,480)}))
cap_right.configure(cap_right.create_preview_configuration(main={"size": (640,480)}))
cap_left.start()
cap_right.start()
sleep(2)
frame_count=0
time_start=time()
while True:
    frame_count+=1
    # ret1, frame_left = cap_left.read()
    # ret2, frame_right = cap_right.read()
    frame_left = cap_left.capture_array()
    frame_right = cap_right.capture_array()

    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
    # if not ret1:
    #     break

    h, w        = frame_left.shape[:2]
    half        = w // 2
    # frame_left  = frame[:, :half]
    # frame_right = frame[:, half:]

    # ── HSV masks ──────────────────────────────────────────────────────────────
    mask_left  = add_HSV_filter(frame_left, 7, MASK_HSV)
    mask_right  = add_HSV_filter(frame_right, 7, MASK_HSV)

    # ── Object detection ───────────────────────────────────────────────────────
    center_right, bbox_right = find_object(frame_right, mask_right)
    center_left,  _          = find_object(frame_left,  mask_left)

    tracking = center_right is not None and center_left is not None

    # ── Depth map + estimation (only when tracking) ────────────────────────────
    depth = None
    if tracking:
        disparity_vis, depth_map = disparity_n_depth_map(mask_left, mask_right, colored=True)
        depth = masked_percentile_depth(depth_map, mask_right, bbox_right)
        if depth is not None:
            depth *=2.5
        
        depth = kf.update(depth)
        print(f"Depth: {depth:.2f} cm" if depth else "Depth: NaN")
         # ── Update matplotlib plot ─────────────────────────
        if depth is not None:

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

            # Update every 5 frames for better FPS
            if frame_count % 5 == 0:

                fig.canvas.draw()
                fig.canvas.flush_events()

    else:

        disparity_vis = np.zeros(
            (480, 640),
            dtype=np.uint8
        )

    # ── Annotate + display ─────────────────────────────────────────────────────
    annotate(frame_right, tracking, depth)
    annotate(frame_left,  tracking, depth)

    cv2.imshow("RIGHT CAMERA", frame_right)
    cv2.imshow("LEFT CAMERA",  frame_left)
    cv2.imshow("Masked Image",  mask_right)

    if cv2.pollKey() & 0xFF == ord('q'):   # non-blocking — saves ~1 ms/frame
        time_diff = time()-time_start
        print(f"\n#| ====================\n#| FRAME DATA\n#|\n#| TOTAL FRAMES: {frame_count}\n#| TOTAL TIME: {time_diff}\n#| FPS: {frame_count/time_diff:.2f}\n#| ====================")
        break

# cap_left.release()
# cap_right.release()
cap_left.stop()
cap_right.stop()
cv2.destroyAllWindows()