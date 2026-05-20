from time import time, sleep
import cv2
import numpy as np
from picamera2 import Picamera2

# ── Stereo / depth constants ───────────────────────────────────────────────────
BASELINE         = 6.5      # camera baseline [cm]
FOV_DEG          = 82       # horizontal FOV [degrees]
FOCAL_LENGTH     = 220.0    # pixels — tune to your camera
DISP_SCALE       = 0.75     # resize factor fed into SGBM (speeds up computation)
HSV_MASK         = [(40, 49, 2), (112, 255, 219)]

# ── Detection constants ────────────────────────────────────────────────────────
MIN_AREA = 50
MORPH_K  = np.ones((5, 5), np.uint8)

# ── SGBM — created once, reused every frame ────────────────────────────────────
_sgbm = cv2.StereoSGBM_create(
    minDisparity      = 0,
    numDisparities    = 16 * 8,   # 128 — must be divisible by 16
    blockSize         = 5,
    P1                = 8  * 3 * 5 ** 2,
    P2                = 32 * 3 * 5 ** 2,
    disp12MaxDiff     = 1,
    uniquenessRatio   = 10,
    speckleWindowSize = 100,
    speckleRange      = 32,
    mode              = cv2.STEREO_SGBM_MODE_SGBM,  # faster than MODE_HH
)


# ── Kalman Filter ──────────────────────────────────────────────────────────────
class KalmanDepthFilter:
    def __init__(self):
        # State vector: [depth, velocity]
        self.x = np.array([[0.0], [0.0]])

        # Covariance matrix
        self.P = np.eye(2) * 1000

        # State transition (dt updated dynamically each frame)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])

        # Measurement model
        self.H = np.array([[1.0, 0.0]])

        # Measurement noise
        self.R = np.array([[10.0]])

        # Process noise
        self.Q = np.array([[0.01, 0.0], [0.0, 0.01]])

        self.initialized = False

    def update(self, measurement, dt: float = 1.0):
        """
        Update the filter with a new measurement.

        Parameters
        ----------
        measurement : float | None
            Raw depth reading. Pass None when object is not detected.
        dt : float
            Elapsed time since last update in seconds (fixes fixed-dt bug).
        """
        # ── Update state-transition matrix with real dt ────────────────────────
        # FIX #5: use actual elapsed time instead of hardcoded 1.0
        self.F[0, 1] = dt

        # ── Prediction step ────────────────────────────────────────────────────
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ── Handle missing measurement ─────────────────────────────────────────
        if measurement is None:
            return float(self.x[0, 0])

        # ── First valid measurement — initialise state ─────────────────────────
        if not self.initialized:
            self.x[0, 0] = measurement
            self.x[1, 0] = 0.0
            self.initialized = True
            return measurement

        # ── Measurement update ─────────────────────────────────────────────────
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])


kf = KalmanDepthFilter()


# ── Core functions ─────────────────────────────────────────────────────────────

def disparity_n_depth_map(
    left:    np.ndarray,
    right:   np.ndarray,
    colored: bool = False,
    scale:   float = DISP_SCALE,
):
    """
    Compute disparity and metric depth map from **grayscale** frames.

    FIX #1 & #2: accepts grayscale images directly (conversion done by caller)
    and the internal commented-out BGR→Gray conversion is removed entirely
    to avoid confusion.

    Parameters
    ----------
    left, right : np.ndarray
        Rectified, single-channel (grayscale) stereo frames.
    colored : bool
        Colourise the disparity visualisation with JET colormap.
    scale : float
        Downscale factor for SGBM speed. 1.0 = full res.

    Returns
    -------
    disp_vis  : np.ndarray  visualisation of disparity
    depth_map : np.ndarray  per-pixel metric depth [cm]
    """
    # Ensure input is truly single-channel
    if left.ndim == 3:
        left  = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    if right.ndim == 3:
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    h, w = left.shape[:2]

    if scale != 1.0:
        dsize = (int(w * scale), int(h * scale))
        left  = cv2.resize(left,  dsize, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, dsize, interpolation=cv2.INTER_LINEAR)

    raw = _sgbm.compute(left, right).astype(np.float32) / 16.0

    if scale != 1.0:
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        raw /= scale  # rescale disparity values back to full resolution

    # ── Depth ──────────────────────────────────────────────────────────────────
    # FIX #4: use tiny epsilon instead of 1.0 offset — only guard true zeros
    valid = raw > 0
    depth = np.zeros_like(raw)
    depth[valid] = (FOCAL_LENGTH * BASELINE) / (raw[valid] + 1e-6)
    depth[~np.isfinite(depth)] = 0

    # ── Disparity visualisation ────────────────────────────────────────────────
    disp_vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if colored:
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    return disp_vis, depth


def hsv_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask isolating the target object via HSV thresholding."""
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_MASK[0], HSV_MASK[1])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_K, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K, iterations=2)
    return mask


def find_object(frame: np.ndarray, mask: np.ndarray):
    """
    Locate the largest masked contour and draw it on *frame*.

    Returns
    -------
    center : (cx, cy) or None
    bbox   : [[x1, y1], [x2, y2]] or None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(c)
    center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 0), -1)

    return center, [[x, y], [x + w, y + h]]


def masked_percentile_depth(
    depth_map:  np.ndarray,
    mask:       np.ndarray,
    bbox,
    percentile: float = 50,
) -> float | None:
    """
    Median (or any percentile) depth inside the object bbox,
    restricted to pixels that are masked, finite, and positive.
    """
    (x1, y1), (x2, y2) = bbox
    values = depth_map[y1:y2, x1:x2][mask[y1:y2, x1:x2] > 0]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, percentile)) if values.size > 0 else None


def annotate(frame: np.ndarray, tracking: bool, depth) -> None:
    """Draw tracking status and depth label onto *frame* in-place."""
    if tracking:
        label = f"Distance: {round(depth, 2) if depth is not None else 'NaN'} cm"
        cv2.putText(frame, "TRACKING", (75,  50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame, label,       (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
    else:
        cv2.putText(frame, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ── Main loop ──────────────────────────────────────────────────────────────────
cap_left  = Picamera2(0)
cap_right = Picamera2(1)
cap_left.configure(cap_left.create_preview_configuration(main={"size": (640, 480)}))
cap_right.configure(cap_right.create_preview_configuration(main={"size": (640, 480)}))
cap_left.start()
cap_right.start()
sleep(2)

frame_count = 0
time_start  = time()
last_time   = time_start  # FIX #5: track per-frame time for Kalman dt

while True:
    frame_count += 1

    # ── Capture ────────────────────────────────────────────────────────────────
    frame_left  = cap_left.capture_array()
    frame_right = cap_right.capture_array()
    frame_left  = cv2.cvtColor(frame_left,  cv2.COLOR_RGB2BGR)
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

    # ── Per-frame dt for Kalman filter ─────────────────────────────────────────
    now = time()
    dt  = now - last_time
    last_time = now

    # ── HSV masks (on colour frames) ───────────────────────────────────────────
    mask_left  = hsv_mask(frame_left)
    mask_right = hsv_mask(frame_right)

    # ── Object detection ───────────────────────────────────────────────────────
    center_right, bbox_right = find_object(frame_right, mask_right)
    center_left,  _          = find_object(frame_left,  mask_left)

    tracking = center_right is not None and center_left is not None

    # ── Depth map + estimation (only when tracking) ────────────────────────────
    depth = None
    if tracking:
        # FIX #1 & #2: convert colour frames to grayscale BEFORE passing to
        # disparity_n_depth_map — NOT binary masks.
        gray_left  = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        _, depth_map = disparity_n_depth_map(gray_left, gray_right, colored=False)

        depth = masked_percentile_depth(depth_map, mask_right, bbox_right)
        # FIX #5: pass real dt to Kalman filter
        depth = kf.update(depth, dt=dt)
        print(f"Depth: {depth:.2f} cm" if depth else "Depth: NaN")

    # ── Annotate + display ─────────────────────────────────────────────────────
    annotate(frame_right, tracking, depth)
    annotate(frame_left,  tracking, depth)

    cv2.imshow("RIGHT CAMERA", frame_right)
    cv2.imshow("LEFT CAMERA",  frame_left)

    if cv2.pollKey() & 0xFF == ord('q'):
        time_diff = time() - time_start
        print(
            f"\n#| ====================\n"
            f"#| FRAME DATA\n"
            f"#|\n"
            f"#| TOTAL FRAMES: {frame_count}\n"
            f"#| TOTAL TIME:   {time_diff:.2f}s\n"
            f"#| FPS:          {frame_count / time_diff:.2f}\n"
            f"#| ===================="
        )
        break

cap_left.stop()
cap_right.stop()
cv2.destroyAllWindows()