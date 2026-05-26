import math
import os

import cv2
import numpy as np
import serial
import time

try:
    from config import ALPHA, FOCAL_LENGTH, BASELINE
except ImportError:
    ALPHA = 70.0
    FOCAL_LENGTH = 600.0
    BASELINE = 10.0

try:
    from config import BLUETOOTH_PORT, BAUD_RATE
except ImportError:
    BLUETOOTH_PORT = "COM5" if os.name == "nt" else "/dev/rfcomm0"
    BAUD_RATE = 115200

ANGLE_MIN = 0
ANGLE_MAX = 180
SERVO_CENTER = 90


def clamp_angle(value):
    return max(ANGLE_MIN, min(ANGLE_MAX, int(round(value))))


def to_servo_angles(horizontal_angle, vertical_angle):
    pan_angle = clamp_angle(SERVO_CENTER + horizontal_angle)
    tilt_angle = clamp_angle(SERVO_CENTER - vertical_angle)
    return pan_angle, tilt_angle


def open_bluetooth_serial():
    try:
        bt_serial = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Bluetooth connected on {BLUETOOTH_PORT} @ {BAUD_RATE}")
        return bt_serial
    except serial.SerialException as exc:
        print(f"Bluetooth unavailable: {exc}")
        return None


def send_angles(bt_serial, pan_angle, tilt_angle):
    if bt_serial is None:
        return

    message = f"{pan_angle},{tilt_angle}\n"
    try:
        bt_serial.write(message.encode("utf-8"))
    except serial.SerialException as exc:
        print(f"Bluetooth send failed: {exc}")


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


def calculate_horizontal_vertical_angles(origin, target, f_pixel):
    # Horizontal offset (positive = right, negative = left)
    delta_x = target[0] - origin[0]
    # Vertical offset (positive = up/above, negative = down/below)
    delta_y = origin[1] - target[1]

    h_angle = math.degrees(math.atan2(delta_x, f_pixel))
    v_angle = math.degrees(math.atan2(delta_y, f_pixel))
    return h_angle, v_angle


cap = cv2.VideoCapture(0)
bt_serial = open_bluetooth_serial()
last_sent_angles = None

while True:
    is_true, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not is_true:
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    # Calculate camera focal length in pixels using horizontal FOV (ALPHA)
    f_pixel = (frame_width * 0.5) / math.tan(math.radians(ALPHA * 0.5))

    # Draw vertical and horizontal central lines (axes)
    cv2.line(frame, (frame_center[0], 0), (frame_center[0], frame_height), (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (0, frame_center[1]), (frame_width, frame_center[1]), (100, 100, 100), 1, cv2.LINE_AA)

    red_mask, red_center, red_contour = find_red_center(frame)

    cv2.circle(frame, frame_center, 5, (255, 255, 0), -1)
    cv2.putText(
        frame,
        "Center",
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

        # Calculate horizontal and vertical angles with sign
        h_angle, v_angle = calculate_horizontal_vertical_angles(frame_center, red_center, f_pixel)
        radial_angle = calculate_radial_angle(frame_center, red_center)
        pan_angle, tilt_angle = to_servo_angles(h_angle, v_angle)

        # Draw a line from frame center to target
        cv2.line(frame, frame_center, red_center, (255, 0, 0), 2)

        # Draw alignment projection lines from target center to the central lines
        cv2.line(frame, red_center, (frame_center[0], red_center[1]), (255, 150, 0), 1, cv2.LINE_AA)
        cv2.line(frame, red_center, (red_center[0], frame_center[1]), (255, 150, 0), 1, cv2.LINE_AA)

        # Assuming static distance between camera and object center is 74 cm (Euclidean Distance D = 74)
        D_static = 74.0
        u_ratio = math.tan(math.radians(h_angle))
        v_ratio = math.tan(math.radians(v_angle))
        
        # Z = D / sqrt(tan^2(h) + tan^2(v) + 1)
        depth = D_static / math.sqrt(u_ratio**2 + v_ratio**2 + 1)
        h_dist = u_ratio * depth
        v_dist = v_ratio * depth

        cv2.putText(
            frame,
            f"Red center: {red_center}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"H-Angle: {h_angle:+.2f} deg",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"V-Angle: {v_angle:+.2f} deg",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Radial Angle: {radial_angle:.2f} deg",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"H-Dist: {h_dist:+.2f} cm",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"V-Dist: {v_dist:+.2f} cm",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Depth (Z): {depth:.2f} cm",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Baseline distance calculation with zero division check
        if abs(h_angle) > 1e-5:
            calculated_distance = BASELINE / (2 * math.tan(math.radians(abs(h_angle))))
            distance_text = f"DISTANCE (Base): {calculated_distance:.2f} cm"
        else:
            distance_text = "DISTANCE (Base): N/A"

        cv2.putText(
            frame,
            distance_text,
            (10, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Pan/Tilt: {pan_angle}, {tilt_angle}",
            (10, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        current_angles = (pan_angle, tilt_angle)
        if current_angles != last_sent_angles:
            send_angles(bt_serial, pan_angle, tilt_angle)
            last_sent_angles = current_angles
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
if bt_serial is not None:
    bt_serial.close()
cv2.destroyAllWindows()

