import cv2
import numpy as np

# ── Stereo / depth constants ───────────────────────────────────────────────────
BASELINE         = 10      # camera baseline [cm]
FOV_DEG          = 68      # horizontal FOV [degrees]
FOCAL_LENGTH     = 600.0   # pixels — tune to your camera
DISPARITY_OFFSET = 1.0     # avoids divide-by-zero at zero disparity
DISP_SCALE       = 0.5     # resize factor fed into SGBM (0.5 = quarter pixels)

# ── Detection constants ────────────────────────────────────────────────────────
MIN_AREA = 300
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

def disparity_n_depth_map(
    left:    np.ndarray,
    right:   np.ndarray,
    colored: bool = False,
    scale:   float = DISP_SCALE,
):
    """
    Compute disparity and metric depth map.

    Speed knobs
    -----------
    scale : resize both frames before SGBM, then upscale result back.
            0.5 => ~4x fewer pixels -> big FPS boost with small quality loss.
    """
    h, w = left.shape[:2]

    # Convert to gray once — SGBM needs single-channel
    gray_l = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        dsize  = (int(w * scale), int(h * scale))
        gray_l = cv2.resize(gray_l, dsize, interpolation=cv2.INTER_LINEAR)
        gray_r = cv2.resize(gray_r, dsize, interpolation=cv2.INTER_LINEAR)

    raw = _sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0

    if scale != 1.0:
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        raw /= scale   # rescale disparity values back to full resolution

    # ── Depth ──────────────────────────────────────────────────────────────────
    valid = raw > 0
    depth = np.zeros_like(raw)
    depth[valid] = (FOCAL_LENGTH * BASELINE) / (raw[valid] + DISPARITY_OFFSET)
    depth[~np.isfinite(depth)] = 0

    # ── Display disparity ──────────────────────────────────────────────────────
    disp_vis = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if colored:
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    return disp_vis, depth


def hsv_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask isolating the target object via HSV thresholding."""
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 80, 50), (179, 255, 255))
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

cap = cv2.VideoCapture(r"assest\car002.m2ts")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w        = frame.shape[:2]
    half        = w // 2
    frame_left  = frame[:, :half]
    frame_right = frame[:, half:]

    # ── HSV masks ──────────────────────────────────────────────────────────────
    mask_left  = hsv_mask(frame_left)
    mask_right = hsv_mask(frame_right)

    # ── Object detection ───────────────────────────────────────────────────────
    center_right, bbox_right = find_object(frame_right, mask_right)
    center_left,  _          = find_object(frame_left,  mask_left)

    tracking = center_right is not None and center_left is not None

    # ── Depth map + estimation (only when tracking) ────────────────────────────
    depth = None
    if tracking:
        _, depth_map = disparity_n_depth_map(frame_left, frame_right, colored=False)
        depth = masked_percentile_depth(depth_map, mask_right, bbox_right)
        print(f"Depth: {depth:.2f} cm" if depth else "Depth: NaN")

    # ── Annotate + display ─────────────────────────────────────────────────────
    annotate(frame_right, tracking, depth)
    annotate(frame_left,  tracking, depth)

    cv2.imshow("RIGHT CAMERA", frame_right)
    cv2.imshow("LEFT CAMERA",  frame_left)

    if cv2.pollKey() & 0xFF == ord('q'):   # non-blocking — saves ~1 ms/frame
        break

cap.release()
cv2.destroyAllWindows()