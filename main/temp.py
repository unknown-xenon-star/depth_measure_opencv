from time import time
import cv2
import numpy as np
from tools.disparity_map import disparity_n_depth_map

kernel = np.ones((5, 5), np.uint8)


def add_HSV_filter(frame):
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 80, 50), (179, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def find_object(frame, mask):
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < 300:
        return None, None

    x, y, w, h = cv2.boundingRect(c)
    center = (x + w // 2, y + h // 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 0), -1)

    return center, ((x, y), (x + w, y + h))


cap = cv2.VideoCapture("assest/video.mp4")

frame_count = 0
start = time()

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    half = w // 2

    frame_left = frame[:, :half]
    frame_right = frame[:, half:]

    mask_left = add_HSV_filter(frame_left)
    mask_right = add_HSV_filter(frame_right)

    center_r, box_r = find_object(frame_right, mask_right)
    center_l, _ = find_object(frame_left, mask_left)

    _, depth_map = disparity_n_depth_map(mask_left, mask_right, True)

    if center_r is None or center_l is None:
        cv2.putText(
            frame_right,
            "TRACKING LOST",
            (75, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:
        (x1, y1), (x2, y2) = box_r

        roi = depth_map[y1:y2, x1:x2]
        roi_mask = mask_right[y1:y2, x1:x2]

        values = roi[roi_mask > 0]
        values = values[np.isfinite(values)]
        values = values[values > 0]

        depth = np.percentile(values, 50) if values.size else None

        cv2.putText(
            frame_right,
            f"Distance: {round(depth, 2) if depth else 'NaN'} cm",
            (200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (124, 252, 0),
            2
        )

        print("Depth:", depth)

    cv2.imshow("RIGHT CAMERA", frame_right)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        elapsed = time() - start

        print(
            f"\nFrames: {frame_count}"
            f"\nTime: {elapsed:.2f}s"
            f"\nFPS: {frame_count / elapsed:.2f}"
        )
        break

cv2.destroyAllWindows()