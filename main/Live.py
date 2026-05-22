from time import time
import cv2
import numpy as np
# import imutils
# import math
from tools.disparity_map import disparity_n_depth_map
from tools.hsv import add_HSV_filter
from tools.detection import find_object, find_depth
from tools.extras import masked_percentile_depth

from tools.config import (
    MODE,

    FOCAL_LENGTH, 
    BASELINE,
    DISPARITY_OFFSET,
    ALPHA,
    MASK_HSV,
    
    LEFT_IMAGE,
    RIGHT_IMAGE 
)



# =========================================================
# MAIN PROGRAM
# =========================================================

# Open left and right cameras
cap = cv2.VideoCapture("assest/video.mp4")
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
    # frame_right = cap_right.copy()
    # frame_left = cap_left.copy()

    # # If capture failed
    # if ret_right == False or ret_left == False:
    #     break

    # =====================================================
    # HSV FILTERING
    # =====================================================

    mask_right = add_HSV_filter(frame_right, 7, MASK_HSV)
    mask_left = add_HSV_filter(frame_left, 7,MASK_HSV)

    # Apply masks
    # res_right = cv2.bitwise_and( frame_right, frame_right, mask=mask_right )
    # res_left = cv2.bitwise_and( frame_left, frame_left, mask=mask_left )

    # =====================================================
    # OBJECT DETECTION
    # =====================================================

    circles_right, corr_r = find_object(frame_right, mask_right)
    circles_left, corr_y = find_object(frame_left, mask_left)
    
    disparity_map, depth_map = disparity_n_depth_map(mask_left, mask_right, True)

    # print(depth_map.shape)
    

    # =====================================================
    # DEPTH CALCULATION
    # =====================================================

    if circles_right is None or circles_left is None:

        cv2.putText( frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText( frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    else:

        # Compute depth
        depth = find_depth( circles_right, circles_left, frame_right, frame_left, BASELINE, ALPHA)
        depth = masked_percentile_depth(depth_map, mask_right, corr_r, 50)
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
    # cv2.imshow("LEFT CAMERA", frame_left)

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
