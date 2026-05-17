import cv2
import numpy as np
import imutils

# =========================================================
# FIND OBJECT CENTER
# =========================================================

def find_object(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold
    _, mask = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = imutils.grab_contours(contours)

    center = None

    if len(contours) > 0:

        # Largest contour
        c = max(contours, key=cv2.contourArea)

        # Minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Moments
        M = cv2.moments(c)

        if M["m00"] != 0:

            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )

            # Draw detected object
            if radius > 3:

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    int(radius),
                    (0,255,0),
                    2
                )

                cv2.circle(
                    frame,
                    center,
                    5,
                    (0,0,255),
                    -1
                )

    return center, mask


# =========================================================
# DEPTH CALCULATION
# =========================================================

def find_depth(
    center_right,
    center_left,
    frame_right,
    frame_left,
    baseline,
    alpha
):

    # Get image width
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    # Check image sizes
    if width_right != width_left:
        print("Images do not have same width")
        return None

    # Focal length in pixels
    f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    # X coordinates
    x_right = center_right[0]
    x_left = center_left[0]

    # Disparity
    disparity = x_left - x_right

    # Avoid division by zero
    if disparity == 0:
        return None

    # Depth formula
    depth = (baseline * f_pixel) / disparity

    return abs(depth)


# =========================================================
# MAIN
# =========================================================

# Baseline distance between cameras [cm]
B = 17.8

# Camera field of view [degrees]
alpha = 72

# cm
DISTANCE = 75

# Load stereo images
frame_left = cv2.imread("/data/l.jpg")
frame_right = cv2.imread("/data/r.jpg")

# Check images
if frame_left is None or frame_right is None:
    print("Error loading images")
    exit()

# =========================================================
# OBJECT DETECTION
# =========================================================

center_left, mask_left = find_object(frame_left)
center_right, mask_right = find_object(frame_right)

print("Left Center:", center_left)
print("Right Center:", center_right)

# =========================================================
# DEPTH ESTIMATION
# =========================================================

if center_left is None or center_right is None:

    print("Object not detected")

    cv2.putText(
        frame_left,
        "TRACKING LOST",
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.putText(
        frame_right,
        "TRACKING LOST",
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

else:

    # Calculate depth
    depth = find_depth(
        center_right,
        center_left,
        frame_right,
        frame_left,
        B,
        alpha
    )

    print("Depth:", round(depth,2), "cm")

    # Display depth
    cv2.putText(
        frame_left,
        f"Depth: {round(depth,2)} cm",
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame_right,
        f"Depth: {round(depth,2)} cm",
        (50,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

# =========================================================
# SHOW WINDOWS
# =========================================================

cv2.imshow("LEFT IMAGE", cv2.resize(frame_left, (0, 0), fx=0.35,fy=0.35))
cv2.imshow("RIGHT IMAGE", cv2.resize(frame_right, (0, 0), fx=0.35,fy=0.35))

cv2.imshow("LEFT MASK", cv2.resize(mask_left, (0, 0), fx=0.35,fy=0.35))
cv2.imshow("RIGHT MASK", cv2.resize(mask_right, (0, 0), fx=0.35,fy=0.35))

print("Press any key to close windows...")

cv2.waitKey(0)

cv2.destroyAllWindows()