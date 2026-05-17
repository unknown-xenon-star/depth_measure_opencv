# depth_map.py

import cv2
import numpy as np

LEFT_IMAGE = "im0.png"
RIGHT_IMAGE = "im1.png"

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
# =========================
# CONFIG
# =========================

FOCAL_LENGTH = 5299.313
BASELINE = 177.288
DISPARITY_OFFSET = 174.186

# =========================
# GLOBALS
# =========================

clicked_x = None
clicked_y = None

current_depth_map = None
current_left_frame = None
current_right_frame = None
current_disparity_map = None

def compute_depth(left_frame, right_frame):
    # Convert to grayscale
    gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32)
    
    # OpenCV scales disparity by 16
    disparity = disparity / 16.0

    # Avoid division by zero
    disparity[disparity <= 0] = 0.1

    # Depth formula
    depth = (FOCAL_LENGTH * BASELINE) / (disparity + DISPARITY_OFFSET)
    

     # Replace invalid depth values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize disparity for display
    disparity_display = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    
    disparity_display = np.uint8(disparity_display)

    # # Apply color map
    # disparity_colormap = cv2.applyColorMap(
    #     disparity_display,
    #     cv2.COLORMAP_JET
    # )

    # return disparity_colormap, depth
    return disparity_display, depth

def mouse_callback(event, x, y, flags, param):

    global clicked_x
    global clicked_y

    global current_depth_map

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_x = x
        clicked_y = y

        depth_value = current_depth_map[y, x]

        print(f"Depth at ({x}, {y}) : {depth_value:.2f} m")

def draw_crosshair(img, x, y, color=(0, 255, 0)):

    size = 15

    cv2.line(img, (x - size, y), (x + size, y), color, 2)
    cv2.line(img, (x, y - size), (x, y + size), color, 2)
    cv2.circle(img, (x, y), 4, color, -1)

def build_display():

    left_display = current_left_frame.copy()
    right_display = current_right_frame.copy()
    depth_display = current_disparity_map.copy()

    if clicked_x is not None and clicked_y is not None:

        depth_value = current_depth_map[clicked_y, clicked_x]

        label = (
            f"Depth: {depth_value:.2f} m"
        )

        draw_crosshair(
            left_display,
            clicked_x,
            clicked_y
        )

        draw_crosshair(
            right_display,
            clicked_x,
            clicked_y
        )

        draw_crosshair(
            depth_display,
            clicked_x,
            clicked_y
        )

        cv2.putText(
            left_display,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    combined = np.hstack([
        left_display,
        right_display,
        depth_display
    ])

    return combined




def main():

    global current_left_frame
    global current_right_frame
    global current_depth_map
    global current_disparity_map

    left = cv2.imread(LEFT_IMAGE)
    right = cv2.imread(RIGHT_IMAGE)

    if left is None or right is None:
        raise FileNotFoundError("Could not load images.")
    
    
    disparity_map, depth_map = compute_depth(
        left,
        right
    )

    current_left_frame = left
    current_right_frame = right
    current_depth_map = depth_map
    current_disparity_map = disparity_map

    cv2.namedWindow("Stereo Depth Viewer")
    cv2.setMouseCallback(
        "Stereo Depth Viewer",
        mouse_callback
    )

    while True:

        display = build_display()

        cv2.imshow(
            "Stereo Depth Viewer",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()