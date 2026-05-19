
import cv2
import numpy as np
from time import time

# =========================================================
# CONFIG
# =========================================================
VIDEO_PATH   = "assest/video.mp4"
BASELINE     = 10
FOCAL_LENGTH = 474
SMOOTHING    = 0.7

# =========================================================
# STEREO MATCHER
# =========================================================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=7,
    P1=8  * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def compute_roi_depth(roi_l, roi_r):
    if roi_l.size == 0 or roi_r.size == 0:
        return None

    if roi_l.shape != roi_r.shape:
        roi_r = cv2.resize(roi_r, (roi_l.shape[1], roi_l.shape[0]))

    g1 = cv2.cvtColor(roi_l, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(g1, g2).astype(np.float32) / 16.0

    valid = disp > 1.0
    if not np.any(valid):
        return None

    depth_map = np.zeros_like(disp)
    depth_map[valid] = (FOCAL_LENGTH * BASELINE) / disp[valid]

    vals = depth_map[valid]
    vals = vals[np.isfinite(vals) & (vals > 0) & (vals < 5000)]

    return float(np.percentile(vals, 10)) if len(vals) > 0 else None


# =========================================================
# OPEN VIDEO
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

ret, first_frame = cap.read()
h, w = first_frame.shape[:2]
left_w = w // 2

# =========================================================
# 0.5× SPEED CONTROL
# =========================================================
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

delay = int((1000 / fps) * 2)  # 0.5x speed

print(f"FPS: {fps}, Delay: {delay} ms (0.5x speed)")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# =========================================================
# STATE
# =========================================================
tracker    = None
locked     = False
prev_depth = None
bbox       = None

frame_count = 0
t_start = time()


# =========================================================
# ROI SELECTOR
# =========================================================
def select_object(frame):
    bbox = cv2.selectROI(
        "STEREO DEPTH",
        frame,
        showCrosshair=True,
        fromCenter=False
    )
    return bbox


# =========================================================
# MAIN LOOP
# =========================================================
while True:

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_depth = None
        frame_count = 0
        continue

    frame_count += 1

    left  = frame[:, :left_w]
    right = frame[:, left_w:]
    display = left.copy()

    # =====================================================
    # SELECT OBJECT (ONCE OR AFTER RESET)
    # =====================================================
    if not locked:
        print("Select object and press ENTER/SPACE")
        init_bbox = select_object(left)

        x, y, bw, bh = init_bbox

        if bw > 0 and bh > 0:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(left, init_bbox)

            locked = True
            bbox = init_bbox

            print(f"Locked on object: {init_bbox}")

    # =====================================================
    # TRACKING
    # =====================================================
    if locked and tracker is not None:

        success, new_bbox = tracker.update(left)

        if not success:
            cv2.putText(display, "TRACK LOST - press R",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            locked = False
            tracker = None

        else:
            x, y, bw, bh = [int(v) for v in new_bbox]

            roi_l = left[y:y+bh, x:x+bw]
            roi_r = right[y:y+bh, x:x+bw]

            depth = compute_roi_depth(roi_l, roi_r)

            if depth is not None:
                if prev_depth is None:
                    prev_depth = depth
                else:
                    prev_depth = SMOOTHING * prev_depth + (1 - SMOOTHING) * depth

            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

            if prev_depth is not None:
                cv2.putText(display, f"{prev_depth:.0f} cm",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                cv2.putText(display, f"{prev_depth/100:.2f} m",
                            (x, y - 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

    else:
        cv2.putText(display,
                    "Drag & select object to track",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2)

    # =====================================================
    # FPS DISPLAY
    # =====================================================
    fps_now = frame_count / max(time() - t_start, 0.001)
    cv2.putText(display, f"FPS: {fps_now:.0f}",
                (left_w - 120, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2)

    cv2.imshow("STEREO DEPTH", display)

    # =====================================================
    # SPEED CONTROL (0.5x)
    # =====================================================
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('r'):
        tracker = None
        locked = False
        prev_depth = None
        print("Reset - select new object")

cap.release()
cv2.destroyAllWindows()