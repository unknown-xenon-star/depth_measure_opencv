import cv2
import numpy as np

from config import (
    MODE,

    FOCAL_LENGTH, 
    BASELINE,
    DISPARITY_OFFSET,
    
    LEFT_IMAGE,
    RIGHT_IMAGE )


# =========================
# SGBM CONFIG
# =========================
stereo = cv2.StereoSGBM_create(
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

def disparity_n_depth_map(
    left:    np.ndarray,
    right:   np.ndarray,
    Colored: bool = False,
    scale:   float = 0.8,

):
    """
    Compute disparity and metric depth map.

    Speed knobs
    -----------
    scale : resize both frames before SGBM, then upscale result back.
            0.5 => ~4x fewer pixels -> big FPS boost with small quality loss.
    """
    h, w = left.shape[:2]

    if scale != 1.0:
        dsize  = (int(w * scale), int(h * scale))
        left = cv2.resize(left, dsize, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, dsize, interpolation=cv2.INTER_LINEAR)
    
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0

    if scale != 1.0:
        disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        disparity /= scale

    # ── Depth ──────────────────────────────────────────────────────────────────
    valid_mask = disparity > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid_mask] = (FOCAL_LENGTH * BASELINE) / (
        disparity[valid_mask] + DISPARITY_OFFSET
    )
    depth[~np.isfinite(depth)] = 0

    # Normalize disparity for display
    disparity_display = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    if Colored:
        # Apply color map
        disparity_display = cv2.applyColorMap(
            disparity_display,
            cv2.COLORMAP_JET
        )


    return disparity_display, depth