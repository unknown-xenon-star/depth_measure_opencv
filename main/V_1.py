import cv2
import numpy as np
import imutils
import math
#177


def resize(img, x):
    return cv2.resize(img, (0,0), fx=x,fy=x)

def add_HSV_filter(frame):

    # Blur image to reduce noise
    blur = cv2.GaussianBlur(frame, (5,5), 0)

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # HSV ranges
    # Change according to your object color

    # Right camera HSV range
    l_b = np.array([60,110,50])
    u_b = np.array([255,255,255])
    
    mask = cv2.inRange(hsv, l_b, u_b)

    # Morphological operations
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask



def find_circles(frame, mask):

    contours = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    contours = imutils.grab_contours(contours)

    center = None

    # If contour exists
    if len(contours) > 0:

        # Largest contour
        c = max(contours, key=cv2.contourArea)

        # Minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Centroid using moments
        M = cv2.moments(c)

        center = (
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        )

        # Ignore tiny objects
        if radius > 10:

            # Draw outer circle
            cv2.circle( frame, (int(x), int(y)), int(radius), (0,255,255), 2 )

            # Draw center point
            cv2.circle( frame, center, 5, (0,0,0), -1 )

    return center




def find_depth( circle_right, circle_left, frame_right, frame_left, baseline, alpha ):

    # Get frame width
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    # Ensure same width
    if width_right != width_left:
        print("Camera frames do not match")
        return None

    # Convert FOV -> focal length in pixels
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    # Extract x coordinates
    x_right = circle_right[0]
    x_left = circle_left[0]

    # Calculate disparity
    disparity = x_left - x_right

    # Avoid division by zero
    if disparity == 0:
        return None

    # Stereo depth formula
    zDepth = (baseline * f_pixel) / disparity

    return abs(zDepth)


# =========================================================
# MAIN PROGRAM
# =========================================================

# Stereo camera baseline distance [cm]
B = 10

# Camera field of view [degrees]
alpha = 68

# Open left and right cameras
cap_right = cv2.imread('assest/im0.png')
cap_left = cv2.imread('assest/im1.png')

while True:

    # Capture frames
    # ret_right, frame_right = cap_right.read()
    # ret_left, frame_left = cap_left.read()
    # ret_right, frame_right = True, cap_right.copy()
    # ret_left, frame_left = True, cap_left.copy()
    frame_right = cap_right.copy()
    frame_left = cap_left.copy()

    # # If capture failed
    # if ret_right == False or ret_left == False:
    #     break

    # =====================================================
    # HSV FILTERING
    # =====================================================

    mask_right = add_HSV_filter(frame_right)
    mask_left = add_HSV_filter(frame_left)

    # Apply masks
    res_right = cv2.bitwise_and( frame_right, frame_right, mask=mask_right )

    res_left = cv2.bitwise_and( frame_left, frame_left, mask=mask_left )

    # =====================================================
    # OBJECT DETECTION
    # =====================================================

    circles_right = find_circles(frame_right, mask_right)
    circles_left = find_circles(frame_left, mask_left)

    # =====================================================
    # DEPTH CALCULATION
    # =====================================================

    if circles_right is None or circles_left is None:

        cv2.putText( frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText( frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    else:

        # Compute depth
        depth = find_depth( circles_right, circles_left, frame_right, frame_left, B, alpha)

        # Show tracking
        cv2.putText( frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2)
        cv2.putText( frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )

        # Show depth
        cv2.putText( frame_right, "Distance: " + str(round(depth,2)) + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )
        cv2.putText( frame_left, "Distance: " + str(round(depth,2)) + " cm", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124,252,0), 2 )

        print("Depth:", depth)

    # =====================================================
    # SHOW WINDOWS
    # =====================================================

    cv2.imshow("RIGHT CAMERA", frame_right)
    cv2.imshow("LEFT CAMERA", frame_left)

    cv2.imshow("MASK RIGHT", mask_right)
    cv2.imshow("MASK LEFT", mask_left)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# =========================================================
# CLEANUP
# =========================================================

# cap_right.release()
# cap_left.release()

cv2.destroyAllWindows()