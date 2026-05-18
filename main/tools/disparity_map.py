import cv2
import numpy as np

from config import (
    MODE,

    FOCAL_LENGTH, 
    BASELINE,
    DISPARITY_OFFSET,
    
    LEFT_IMAGE,
    RIGHT_IMAGE 
)


# =========================
# SGBM CONFIG
# =========================
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 12,   # must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

def disparity_n_depth_map(left, right, Colored=False):
    
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0

    # Mask invalid disparity
    valid_mask = disparity > 0

    depth = np.zeros_like(disparity, dtype=np.float32)

    #  Use correct formula (choose one)
    depth[valid_mask] = (FOCAL_LENGTH * BASELINE) / (
        disparity[valid_mask] + DISPARITY_OFFSET
    )

    # Clean invalid values
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