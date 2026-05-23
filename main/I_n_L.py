import numpy as np
import cv2
from time import time
from tools.hsv import add_HSV_filter
from config import MASK_HSV

# =======================================
# CONFIG
# =======================================
# ── Stereo / depth constants ───────────────────────────────────────────────────
BASELINE         = 10      # camera baseline [cm]
FOV_DEG          = 68      # horizontal FOV [degrees]
FOCAL_LENGTH     = 600.0   # pixels — tune to your camera
DISPARITY_OFFSET = 1.0     # avoids divide-by-zero at zero disparity
DISP_SCALE       = 0.8     # resize factor fed into SGBM (0.5 = quarter pixels)

# ── Detection constants ────────────────────────────────────────────────────────
MIN_AREA = 300
MORPH_K  = np.ones((5, 5), np.uint8)
# =======================================
# GLOBALS
# =======================================
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


    if scale != 1.0:
        dsize  = (int(w * scale), int(h * scale))
        left = cv2.resize(left, dsize, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, dsize, interpolation=cv2.INTER_LINEAR)

    disparity = stereo.compute(left, right).astype(np.float32) / 16.0

    if scale != 1.0:
        disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        disparity /= scale   # rescale disparity values back to full resolution

    # ── Depth ──────────────────────────────────────────────────────────────────
    valid_mask = disparity > 0
    depth = np.zeros_like(disparity)
    depth[valid_mask] = (FOCAL_LENGTH * BASELINE) / (disparity[valid_mask] + DISPARITY_OFFSET)
    depth[~np.isfinite(depth)] = 0

    # ── Display disparity ──────────────────────────────────────────────────────
    disparity_display = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if colored:
        disparity_display = cv2.applyColorMap(disparity_display, cv2.COLORMAP_JET)

    return disparity_display, depth

def find_object(frame, mask):

    contours, _ = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    # center = None
    if not contours:
        return None, None

    # Largest contour
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 300:
        return None, None

    x, y, w, h = cv2.boundingRect(c)
    center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0,0,0), -1)

    return center, [[x, y], [x+w, y+h]]

def main():

    # Open left and right cameras
    cap = cv2.VideoCapture("assest\car002.m2ts")
    frame_count=0
    time_start=time()
    while True:
        frame_count+=1
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        half = w // 2

        frame_left = frame[:, :half]
        frame_right = frame[:, half:]

        # =====================================================
        # HSV FILTERING
        # =====================================================
        mask_left  = add_HSV_filter(frame_left, 7, MASK_HSV)
        mask_right  = add_HSV_filter(frame_right, 7, MASK_HSV)

        # Apply masks
        # res_right = cv2.bitwise_and( frame_right, frame_right, mask=mask_right )
        # res_left = cv2.bitwise_and( frame_left, frame_left, mask=mask_left )

        # =====================================================
        # OBJECT DETECTION
        # =====================================================
        circles_right, corr_r = find_object(frame_right, mask_right)
        circles_left, corr_y = find_object(frame_left, mask_left)
        
        # =====================================================
        # DEPTH CALCULATION
        # =====================================================
        disparity_map, depth_map = disparity_n_depth_map(frame_left, frame_right, True)
        
        if circles_right is None or circles_left is None:

            cv2.putText( frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText( frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        else:

            # Compute depth
            # depth = find_depth( circles_right, circles_left, frame_right, frame_left, B, alpha)
            # depth_ = depth_map 
            roi = depth_map[ corr_r[0][1]:corr_r[1][1], corr_r[0][0]:corr_r[1][0] ]

            roi_mask = mask_right[corr_r[0][1]:corr_r[1][1],corr_r[0][0]:corr_r[1][0]]

            values = roi[roi_mask > 0]
            values = values[np.isfinite(values)]
            values = values[values > 0]
            if values.size > 0:
                depth = np.percentile(values, 50)
            else:
                depth = None
            # Show tracking
            cv2.putText( frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2)
            cv2.putText( frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )

            # Show depth
            cv2.putText( frame_right, "Distance: " + str(round(depth,2) if depth else "NaN" ) + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )
            cv2.putText( frame_left, "Distance: " + str(round(depth,2) if depth else "NaN") + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )

            print("Depth:", depth)

        # =====================================================
        # SHOW WINDOWS
        # =====================================================
        cv2.imshow("RIGHT CAMERA", frame_right)
        cv2.imshow("LEFT CAMERA", frame_left)

        # cv2.imshow("MASK RIGHT", mask_right)
        # cv2.imshow("MASK LEFT", mask_left)

        # cv2.imshow("DEPTH MAP",depth_map)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            time_diff = time()-time_start
            print(f"\n#| ====================\n#| FRAME DATA\n#|\n#| TOTAL FRAMES: {frame_count}\n#| TOTAL TIME: {time_diff}\n#| FPS: {frame_count/time_diff:.2f}\n#| ====================")
            break


    # =========================================================
    # CLEANUP
    # =========================================================
    # cap_right.release()
    # cap_left.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()