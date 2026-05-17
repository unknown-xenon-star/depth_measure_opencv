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


def disparity_n_depth_map(left, right, colored=False):

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

    disparity = stereo.compute(left, right).astype(np.float32)
    
    # OpenCV scales disparity by 16
    disparity = disparity / 16.0

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
    )
    
    disparity_display = np.uint8(disparity_display)

    if colored:
        # Apply color map
        disparity_colormap = cv2.applyColorMap(
            disparity_display,
            cv2.COLORMAP_JET
        )
        
        return disparity_colormap, depth

    return disparity_display, depth