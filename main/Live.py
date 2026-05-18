from time import time
import cv2
import numpy as np
# import imutils
# import math
# from tools.object import cropped
from tools.disparity_map import disparity_n_depth_map
#177


def resize(img, x):
    return cv2.resize(img, (0,0), fx=x,fy=x)



# def add_HSV_filter(frame):

#     # Reduce noise + preserve edges better than plain blur
#     blur = cv2.GaussianBlur(frame, (7,7), 0)

#     # Convert BGR -> HSV
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#     # More flexible HSV range (wider, works for more objects)
#     # You can still tune these if needed
#     l_b = np.array([0, 80, 50])
#     u_b = np.array([179, 255, 255])

#     mask = cv2.inRange(hsv, l_b, u_b)

#     # Clean small noise
#     kernel = np.ones((5,5), np.uint8)

#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     return mask

kernel = np.ones((5,5), np.uint8)

def add_HSV_filter(frame):

    # Blur image to reduce noise
    blur = cv2.GaussianBlur(frame, (7,7), 0)

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # HSV ranges
    # Change according to your object color

    # Right camera HSV range
    mask = cv2.inRange(
        hsv,
        (0, 80, 50),
        (179, 255, 255)
    )
    
    # kernel = np.ones((5,5), np.uint8)

    # Morphological operations
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def get_roi(frame, model):
    """Return cropped ROI using YOLO, or full frame if no detection."""
    boxes = cropped(frame, model=model)
    if not boxes:
        return frame, None
    x1, y1, x2, y2 = boxes[0]   # first match
    return frame[y1:y2, x1:x2], (x1, y1)

def find_object(frame, mask):

    contours, _ = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # contours = imutils.grab_contours(contours)

    # center = None
    if not contours:
        return None, None

    # If contour exists
    # if len(contours) > 0:

    # Largest contour
    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 300:
        return None, None

    x, y, w, h = cv2.boundingRect(c)

    center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.circle(frame, center, 5, (0,0,0), -1)

    return center, [[x, y], [x+w, y+h]]

def find_depth( circle_right, circle_left, frame_right, frame_left, baseline, alpha, depth_map=0):

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
    # frame_right = cap_right.copy()
    # frame_left = cap_left.copy()

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

    circles_right, corr_r = find_object(frame_right, mask_right)
    circles_left, corr_y = find_object(frame_left, mask_left)
    
    disparity_map, depth_map = disparity_n_depth_map(frame_left, frame_right, True)

    # print(depth_map.shape)
    

    # =====================================================
    # DEPTH CALCULATION
    # =====================================================

    if circles_right is None or circles_left is None:

        cv2.putText( frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText( frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    else:

        # Compute depth
        depth = find_depth( circles_right, circles_left, frame_right, frame_left, B, alpha)
        # depth_ = depth_map 
        roi = depth_map[
            corr_r[0][1]:corr_r[1][1],
            corr_r[0][0]:corr_r[1][0]
        ]

        roi_mask = mask_right[
            corr_r[0][1]:corr_r[1][1],
            corr_r[0][0]:corr_r[1][0]
        ]

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

    # cv2.imshow("RIGHT CAMERA", frame_right)
    # cv2.imshow("LEFT CAMERA", frame_left)

    cv2.imshow("MASK RIGHT", mask_right)
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