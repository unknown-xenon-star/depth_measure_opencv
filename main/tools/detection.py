import cv2
import numpy as np
from config import MIN_AREA

def crop_cord(x,y,w,h, depth_map):
    cx = x + w//4
    cy = y + h//4

    cw = w//2
    ch = h//2

    center_depth = depth_map[cy:cy+ch, cx:cx+cw]
    object_depth = depth_map[y:y+h, x:x+w]


    # Remove invalid values
    # valid = object_depth[np.isfinite(object_depth)]
    valid = object_depth[np.isfinite(object_depth)]
    valid = valid[valid > 0]

    if len(valid) > 0:
        avg_depth = np.mean(valid)
        median_depth = np.median(valid)

        print("Average depth:", avg_depth)
        print("Median depth:", median_depth)
    else:
        print("No valid depth values found")

def segmentation(depth_map, contour_mask):

    mask = np.zeros(depth_map.shape, dtype=np.uint8)

    cv2.drawContours(mask, [contour_mask], -1, 255, -1)

    object_pixels = depth_map[mask == 255]

    valid = object_pixels[np.isfinite(object_pixels)]
    valid = valid[valid > 0]

    avg_depth = np.mean(valid)
    median_depth = np.median(valid)
    print(median_depth)

def find_circles(frame, mask):

    contours_data = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Compatible with OpenCV 3 and 4
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

    center = None
    radius = 0

    # If contour exists
    if len(contours) > 0:

        # Largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Centroid using moments
        moments = cv2.moments(largest_contour)

        if moments["m00"] != 0:
            center = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"])
            )

            # Ignore tiny objects
            if radius > 10:

                # Draw outer circle
                cv2.circle( frame, (int(x), int(y)), int(radius), (0, 255, 255), 1 )
                # Draw center point
                cv2.circle( frame, center, 5, (0, 0, 0), -1 )

    return center, radius

def find_object(frame: np.ndarray, mask: np.ndarray):
    """
    Locate the largest masked contour and draw it on *frame*.

    Returns
    -------
    center : (cx, cy) or None
    bbox   : [[x1, y1], [x2, y2]] or None
    """

    contours, _ = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # contours = imutils.grab_contours(contours)

    # center = None
    if not contours:
        return None, None

    # If contour exists
    # if len(contours) > 0:

    # Largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < MIN_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(largest_contour)

    center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.circle(frame, center, 5, (0,0,0), -1)

    return center, [[x, y], [x+w, y+h]]

def find_depth(
    circle_right,
    circle_left,
    frame_right,
    frame_left,
    baseline,
    ALPHA
):

    # Get frame width
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    # Ensure same width
    if width_right != width_left:
        print("Camera frames do not match")
        return None

    # Convert FOV -> focal length in pixels
    f_pixel = (width_right * 0.5) / np.tan(ALPHA * 0.5 * np.pi / 180)

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
