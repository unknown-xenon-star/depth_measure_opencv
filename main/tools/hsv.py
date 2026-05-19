import cv2
import numpy as np
import imutils
import math
from config import BASELINE, ALPHA, MASK_HSV

MORPH_K = np.ones((5,5), np.uint8)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.detection import find_circles, find_depth, find_object

def add_HSV_filter(frame: np.ndarray, blur_factor, mask) -> np.ndarray:
    """Binary mask isolating the target object via HSV thresholding."""

    # Blur image to reduce noise
    blur = cv2.GaussianBlur(frame, (blur_factor, blur_factor), 0)

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # HSV ranges
    # Change according to your object color
    mask = cv2.inRange(
        hsv,
        mask[0],
        mask[1]
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_K, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K, iterations=2)

    return mask


# =========================================================
# MAIN PROGRAM
# =========================================================

# Stereo camera baseline distance [cm]


def main():
    # Open left and right cameras
    cap_right = cv2.imread('data/l.jpg')
    cap_left = cv2.imread('data/r.jpg')

    while True:

        # Capture frames
        frame_right = cap_right.copy()
        frame_left = cap_left.copy()

        # # If capture failed
        # if ret_right == False or ret_left == False:
        #     break

        # =====================================================
        # HSV FILTERING
        # =====================================================

        mask_right = add_HSV_filter(frame_right, 5, MASK_HSV)
        mask_left = add_HSV_filter(frame_left, 5, MASK_HSV)

        # Apply masks
        # res_right = cv2.bitwise_and(
        #     frame_right,
        #     frame_right,
        #     mask=mask_right
        # )

        # res_left = cv2.bitwise_and(
        #     frame_left,
        #     frame_left,
        #     mask=mask_left
        # )

        # =====================================================
        # OBJECT DETECTION
        # =====================================================

        circles_right,_ = find_object(frame_right, mask_right)
        circles_left,_ = find_object(frame_left, mask_left)

        # =====================================================
        # DEPTH CALCULATION
        # =====================================================

        if circles_right is None or circles_left is None:

            cv2.putText( frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2 )
            cv2.putText( frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2 )

        else:

            # Compute depth
            depth = find_depth( circles_right, circles_left, frame_right, frame_left, BASELINE, ALPHA )*74/41.7

            # Show tracking
            cv2.putText( frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )
            cv2.putText( frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )
            # Show depth
            cv2.putText( frame_right, "Distance: " + str(round(depth,2)) + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )
            cv2.putText( frame_left, "Distance: " + str(round(depth,2)) + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )

            print("Depth:", depth)

        # =====================================================
        # SHOW WINDOWS
        # =====================================================

        cv2.imshow("RIGHT CAMERA", frame_right)
        # cv2.imshow("LEFT CAMERA", frame_left)

        cv2.imshow("MASK RIGHT", mask_right)
        # cv2.imshow("MASK LEFT", mask_left)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # =========================================================
    # CLEANUP
    # =========================================================

    # cap_right.release()
    # cap_left.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()