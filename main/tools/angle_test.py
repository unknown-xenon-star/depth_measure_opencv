import math

import cv2
import numpy as np


def find_red_center(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around the HSV hue range, so use two bands.
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return red_mask, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 300:
        return red_mask, None, None

    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return red_mask, None, None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return red_mask, (center_x, center_y), largest_contour


def calculate_radial_angle(origin, target):
    delta_x = target[0] - origin[0]
    delta_y = origin[1] - target[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle


cap = cv2.VideoCapture(0)

while True:
    is_true, frame = cap.read()

    if not is_true:
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    red_mask, red_center, red_contour = find_red_center(frame)

    cv2.circle(frame, frame_center, 5, (255, 255, 0), -1)
    cv2.putText(
        frame,
        "Frame center",
        (frame_center[0] + 10, frame_center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    if red_center is not None and red_contour is not None:
        cv2.drawContours(frame, [red_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, red_center, 6, (0, 0, 255), -1)

        angle = calculate_radial_angle(frame_center, red_center)

        cv2.line(frame, frame_center, red_center, (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"Red center: {red_center}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Angle: {angle:.2f} deg",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "No red object detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("frame", frame)
    cv2.imshow("red mask", red_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

