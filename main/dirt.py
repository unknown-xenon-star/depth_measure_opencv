"""
tracker_with_vpython.py
-----------------------
Red-object tracker + BNO055 3-D visualisation (VPython) + ultrasonic distance.

OpenCV runs in a background thread while VPython handles the main thread.
Distance (cm) received from the ESP32 ultrasonic sensor is shown on the frame.

STEREO INTEGRATION:
- Dual Picamera2(0) + Picamera2(1) stereo input                [UNCHANGED - already present]
- StereoSGBM disparity map                                      [NEW]
- Hybrid K-Means + GrabCut segmentation mask                    [NEW]
- HSV filtering as segmentation hint                            [NEW]
- Kalman depth filter                                           [NEW]
- True stereo depth replaces fake D_static/tan() monocular      [NEW]
- Disparity + Hybrid Mask windows                               [NEW]
"""

import math
import os
import threading
import time

import cv2
import numpy as np

# ── Stereo / hybrid segmentation pipeline ────────────────────────────────────
from tools.kalman_filter       import KalmanDepthTracker
from tools.disparity_map       import disparity_n_depth_map
from tools.hsv                 import add_HSV_filter
from tools.extras              import masked_percentile_depth
from tools.detection           import find_object
from tools.hybrid_segmentation import hybrid_mask
import serial
from picamera2 import Picamera2

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import ALPHA, FOCAL_LENGTH, BASELINE
except ImportError:
    ALPHA        = 70.0
    FOCAL_LENGTH = 600.0
    BASELINE     = 10.0

try:
    from config import BLUETOOTH_PORT, BAUD_RATE
except ImportError:
    BLUETOOTH_PORT = "COM5" if os.name == "nt" else "/dev/rfcomm0"
    BAUD_RATE      = 115200

# ── Servo constants ───────────────────────────────────────────────────────────
ANGLE_MIN    = 0
ANGLE_MAX    = 180
SERVO_CENTER = 90
PAN_OFFSET   = 0
TILT_OFFSET  = 10
TILT_HOME    = 90
PAN_HOME     = 90

YAW_OFFSET_DEG = -45.0

# ── Sensor-Camera Spatial Offset ──────────────────────────────────────────────
CAM_BEHIND_SENSOR_CM  = 10.0
CAM_ABOVE_SENSOR_CM   =  7.0

# ── Stereo depth constants ────────────────────────────────────────────────────
STEREO_BASELINE    = 6.5   # cm  — physical lens-to-lens distance, tune to rig
STEREO_DEPTH_SCALE = 2.5   # empirical correction factor, tune on bench
HYBRID_INTERVAL    = 1     # 0=HSV only  1=hybrid every frame  N=hybrid every N frames

# HSV hint range — matches the orange/red target that find_red_center tracks
MASK_HSV = [(15, 80, 40), (40, 255, 255)]

# ── SGBM — created once at module level, reused every frame ──────────────────
_sgbm = cv2.StereoSGBM_create(
    minDisparity      = 0,
    numDisparities    = 16 * 8,
    blockSize         = 5,
    P1                = 8  * 3 * 5 ** 2,
    P2                = 32 * 3 * 5 ** 2,
    disp12MaxDiff     = 1,
    uniquenessRatio   = 10,
    speckleWindowSize = 100,
    speckleRange      = 32,
    mode              = cv2.STEREO_SGBM_MODE_SGBM,
)

# ── Shared state (written by BT reader, read by VPython & CV loops) ──────────
shared = {
    "quat":        [1.0, 0.0, 0.0, 0.0],   # w, x, y, z
    "calib":       [0, 0, 0, 0],            # sys, gyro, accel, mg
    "distance":    None,                    # ultrasonic cm (float or None)
    "stereo_depth": None,                   # true stereo depth cm (float or None)
    "yaw_zero":    0.0,
    "pitch_zero":  0.0,
    "current_yaw": 0.0,
    "current_pitch": 0.0,
    "running":     True,
    "lock":        threading.Lock(),
}

# ═════════════════════════════════════════════════════════════════════════════
#  Bluetooth & Vision helpers  — ALL UNCHANGED
# ═════════════════════════════════════════════════════════════════════════════
def clamp_angle(value):
    return max(ANGLE_MIN, min(ANGLE_MAX, int(round(value))))


def to_servo_angles(horizontal_angle, vertical_angle):
    pan  = clamp_angle(PAN_HOME  + horizontal_angle + PAN_OFFSET)
    tilt = clamp_angle(TILT_HOME + vertical_angle   + TILT_OFFSET)
    return pan, tilt


def open_bluetooth_serial():
    try:
        bt = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=0)
        time.sleep(2)
        print(f"Bluetooth connected on {BLUETOOTH_PORT} @ {BAUD_RATE}")
        return bt
    except serial.SerialException as exc:
        print(f"Bluetooth unavailable: {exc}")
        return None


def send_angles(bt_serial, pan_angle, tilt_angle):
    if bt_serial is None:
        return
    try:
        bt_serial.write(f"{pan_angle},{tilt_angle}\n".encode("utf-8"))
    except serial.SerialException as exc:
        print(f"Bluetooth send failed: {exc}")


def read_imu_data(bt_serial):
    if bt_serial is None:
        return None

    latest = None
    try:
        while bt_serial.in_waiting > 0:
            raw   = bt_serial.readline().decode("utf-8", errors="replace").strip()
            if not raw.startswith("IMU,"):
                continue
            parts = raw.split(",")
            if len(parts) < 9:
                continue

            try:
                w, x, y, z = (float(parts[i]) for i in range(1, 5))
                accel_c = int(parts[5])
                gyro_c  = int(parts[6])
                mg_c    = int(parts[7])
                sys_c   = int(parts[8])
                dist    = float(parts[9]) if len(parts) >= 10 else None
            except (ValueError, IndexError):
                continue

            with shared["lock"]:
                shared["quat"]     = [w, x, y, z]
                shared["calib"]    = [sys_c, gyro_c, accel_c, mg_c]
                shared["distance"] = dist

            latest = [f"{w:.4f}", f"{z:.4f}", f"{y:.4f}", f"{x:.4f}"]
    except Exception:
        pass

    return latest


def find_red_center(frame):
    """UNCHANGED — original HSV contour detector kept for angle/contour drawing."""
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, np.array([15,  80,  40]), np.array([35, 255, 180]))
    mask_2 = cv2.inRange(hsv, np.array([18,  70, 120]), np.array([40, 255, 255]))
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    kernel   = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return red_mask, None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return red_mask, None, None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return red_mask, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return red_mask, (cx, cy), largest


def calculate_horizontal_vertical_angles(origin, target, f_pixel):
    dx = target[0] - origin[0]
    dy = origin[1] - target[1]
    return (math.degrees(math.atan2(dx, f_pixel)),
            math.degrees(math.atan2(dy, f_pixel)))


def calculate_radial_angle(origin, target):
    return math.degrees(math.atan2(origin[1] - target[1], target[0] - origin[0]))


# ═════════════════════════════════════════════════════════════════════════════
#  OpenCV loop  (background thread)
# ═════════════════════════════════════════════════════════════════════════════
def opencv_thread():
    global CAM_ABOVE_SENSOR_CM, CAM_BEHIND_SENSOR_CM

    # ── Camera init — UNCHANGED ───────────────────────────────────────────────
    left_cam  = Picamera2(0)
    right_cam = Picamera2(1)

    left_cam.configure(
        left_cam.create_preview_configuration(main={"size": (640, 480)})
    )
    right_cam.configure(
        right_cam.create_preview_configuration(main={"size": (640, 480)})
    )

    left_cam.start()
    right_cam.start()
    time.sleep(2)

    bt_serial = open_bluetooth_serial()

    # ── Per-thread state ──────────────────────────────────────────────────────
    last_sent_angles   = None
    last_imu_display   = ["0.0000", "0.0000", "0.0000", "0.0000"]
    manual_pan_offset  = 0
    manual_tilt_offset = 0

    # ── Stereo depth pipeline state ───────────────────────────────────────────
    kf                 = KalmanDepthTracker()
    _cached_mask_left  = None
    _cached_mask_right = None
    frame_count        = 0

    # ── Main capture loop ─────────────────────────────────────────────────────
    while True:
        with shared["lock"]:
            if not shared["running"]:
                break

        frame_count += 1

        # ── Capture + colour convert — UNCHANGED ─────────────────────────────
        frame_left  = left_cam.capture_array()
        frame_right = right_cam.capture_array()

        cv2.flip(frame_left,  0, dst=frame_left)
        cv2.flip(frame_right, 0, dst=frame_right)
        cv2.flip(frame_left,  1, dst=frame_left)
        cv2.flip(frame_right, 1, dst=frame_right)

        frame_left  = cv2.cvtColor(frame_left,  cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

        frame = frame_right.copy()   # display frame = right camera (UNCHANGED)

        # ── Keyboard handling — UNCHANGED ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            with shared["lock"]:
                shared["running"] = False
            break
        elif key == ord("w"):
            manual_tilt_offset += 3
            print(f"Manual Servo Command Nudge -> Tilt Up (Offset: {manual_tilt_offset})")
        elif key == ord("s"):
            manual_tilt_offset -= 3
            print(f"Manual Servo Command Nudge -> Tilt Down (Offset: {manual_tilt_offset})")
        elif key == ord("a"):
            manual_pan_offset -= 3
            print(f"Manual Servo Command Nudge -> Pan Left (Offset: {manual_pan_offset})")
        elif key == ord("d"):
            manual_pan_offset += 3
            print(f"Manual Servo Command Nudge -> Pan Right (Offset: {manual_pan_offset})")
        elif key == ord("r"):
            manual_pan_offset  = 0
            manual_tilt_offset = 0
            print("Manual Servo offsets reset to home center.")
        elif key == ord("f"):
            with shared["lock"]:
                shared["yaw_zero"] += shared["current_yaw"]
            print("Yaw reset to 0.")
        elif key == ord("g"):
            with shared["lock"]:
                shared["pitch_zero"] += shared["current_pitch"]
            print("Pitch reset to 0.")

        # ── Frame geometry — UNCHANGED ────────────────────────────────────────
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        center = (w // 2, h // 2)
        f_px   = (w * 0.5) / math.tan(math.radians(ALPHA * 0.5))

        # Cross-hair — UNCHANGED
        cv2.line(frame, (center[0], 0), (center[0], h), (100, 100, 100), 1)
        cv2.line(frame, (0, center[1]), (w, center[1]), (100, 100, 100), 1)

        # ── IMU read + display — UNCHANGED ────────────────────────────────────
        imu_data = read_imu_data(bt_serial)
        if imu_data is not None:
            last_imu_display = imu_data

        w_v, x_v, y_v, z_v = last_imu_display
        cv2.putText(frame, f"Quat W={w_v} X={x_v} Y={y_v} Z={z_v}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 255), 2)

        with shared["lock"]:
            dist_cm = shared["distance"]
            sys_c, gyr_c, acc_c, mg_c = shared["calib"]

        dist_text  = f"Ultrasonic: {dist_cm:.1f} cm" if dist_cm is not None else "Ultrasonic: --"
        dist_color = (0, 255, 255) if dist_cm is not None else (100, 100, 100)
        cv2.putText(frame, dist_text,
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_color, 2)
        cv2.putText(frame, f"Calib Sys={sys_c} Gyr={gyr_c} Acc={acc_c} Mg={mg_c}",
                    (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        # ═════════════════════════════════════════════════════════════════════
        #  STEREO PIPELINE  — replaces D_static / tan() monocular depth
        # ═════════════════════════════════════════════════════════════════════

        # ── Step 1: Hybrid segmentation masks (K-Means + GrabCut + HSV hint) ─
        if HYBRID_INTERVAL == 0:
            # Pure HSV — fastest, no K-Means/GrabCut
            mask_left  = add_HSV_filter(frame_left,  7, MASK_HSV)
            mask_right = add_HSV_filter(frame_right, 7, MASK_HSV)

        elif frame_count % HYBRID_INTERVAL == 1 or _cached_mask_left is None:
            # Full hybrid pass: normalise → L*a*b* → adaptive K-Means →
            #                   auto GrabCut → shape refinement → morphology
            mask_left  = hybrid_mask(frame_left,  blur_k=7, hsv_range=MASK_HSV)
            mask_right = hybrid_mask(frame_right, blur_k=7, hsv_range=MASK_HSV)
            _cached_mask_left  = mask_left.copy()
            _cached_mask_right = mask_right.copy()

        else:
            # Fast inter-frame update: fresh HSV AND cached hybrid mask
            hsv_l = add_HSV_filter(frame_left,  7, MASK_HSV)
            hsv_r = add_HSV_filter(frame_right, 7, MASK_HSV)
            mask_left  = cv2.bitwise_and(_cached_mask_left,  hsv_l)
            mask_right = cv2.bitwise_and(_cached_mask_right, hsv_r)
            # Fall back to cache if fast update wiped the mask
            if np.sum(mask_left)  < 500:
                mask_left  = _cached_mask_left.copy()
            if np.sum(mask_right) < 500:
                mask_right = _cached_mask_right.copy()

        # ── Step 2: Stereo object detection ──────────────────────────────────
        center_right, bbox_right = find_object(frame_right, mask_right)
        center_left,  _          = find_object(frame_left,  mask_left)
        stereo_tracking = center_right is not None and center_left is not None

        # ── Step 3: Disparity map + true stereo depth ─────────────────────────
        stereo_depth  = None
        disparity_vis = np.zeros((h, w), dtype=np.uint8)

        if stereo_tracking:
            disparity_vis, depth_map = disparity_n_depth_map(
                mask_left, mask_right, colored=True
            )
            raw_depth = masked_percentile_depth(depth_map, mask_right, bbox_right)
            if raw_depth is not None:
                raw_depth *= STEREO_DEPTH_SCALE
            stereo_depth = kf.update(raw_depth)

        # Write stereo depth into shared state so VPython can use it
        with shared["lock"]:
            shared["stereo_depth"] = stereo_depth

        # ═════════════════════════════════════════════════════════════════════
        #  OBJECT DETECTION + SERVO CONTROL
        #  find_red_center gives contour + angle; stereo depth replaces D_static
        # ═════════════════════════════════════════════════════════════════════
        red_mask, red_center, red_contour = find_red_center(frame)
        cv2.circle(frame, center, 5, (255, 255, 0), -1)

        pan_angle, tilt_angle = to_servo_angles(manual_pan_offset, manual_tilt_offset)

        if red_center is not None and red_contour is not None:
            cv2.drawContours(frame, [red_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, red_center, 6, (0, 0, 255), -1)

            h_angle, v_angle = calculate_horizontal_vertical_angles(
                center, red_center, f_px
            )
            pan_angle, tilt_angle = to_servo_angles(
                h_angle + manual_pan_offset,
                v_angle + manual_tilt_offset
            )
            cv2.line(frame, center, red_center, (255, 0, 0), 2)

            # ── TRUE STEREO DEPTH (D_static/tan() removed) ───────────────────
            if stereo_depth is not None:
                depth_label = f"Stereo Depth: {stereo_depth:.1f} cm"
                depth_color = (0, 255, 255)
            else:
                depth_label = "Stereo Depth: --"
                depth_color = (100, 100, 100)

            cv2.putText(frame, depth_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_color, 2)
            cv2.putText(frame, f"Pan/Tilt: {pan_angle},{tilt_angle}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Tracking | Stereo OK: {stereo_tracking}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        else:
            cv2.putText(frame, "No object - Manual Drive Active",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Pan/Tilt: {pan_angle},{tilt_angle}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ── Servo send — UNCHANGED logic ──────────────────────────────────────
        if (pan_angle, tilt_angle) != last_sent_angles:
            send_angles(bt_serial, pan_angle, tilt_angle)
            last_sent_angles = (pan_angle, tilt_angle)

        # ── Display windows ───────────────────────────────────────────────────
        cv2.imshow("Tracker",      frame)
        cv2.imshow("Red Mask",     red_mask)
        cv2.imshow("Hybrid Mask",  mask_right)
        cv2.imshow("Disparity",    disparity_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            with shared["lock"]:
                shared["running"] = False
            break

    # ── Cleanup — UNCHANGED ───────────────────────────────────────────────────
    left_cam.stop()
    right_cam.stop()
    if bt_serial:
        bt_serial.close()
    cv2.destroyAllWindows()


# ═════════════════════════════════════════════════════════════════════════════
#  Main loop  (main thread — required by VPython)   ALL UNCHANGED
# ═════════════════════════════════════════════════════════════════════════════
def main():
    from vpython import arrow, box, color, compound, cross, label, rate, scene, vector, sphere

    # ── Scene setup — UNCHANGED ───────────────────────────────────────────────
    scene.title      = "BNO055 Orientation (Origin @ Ultrasonic Sensor)"
    scene.range      = 5
    scene.forward    = vector(-1, -1, -1)
    scene.width      = 600
    scene.height     = 600
    scene.background = color.black

    arrow(length=2, shaftwidth=.1, color=color.red,   axis=vector(1, 0, 0))
    arrow(length=2, shaftwidth=.1, color=color.green, axis=vector(0, 1, 0))
    arrow(length=4, shaftwidth=.1, color=color.blue,  axis=vector(0, 0, 1))

    front_arrow = arrow(length=1.0, shaftwidth=.1, color=color.purple,  axis=vector(1, 0, 0))
    up_arrow    = arrow(length=1,   shaftwidth=.1, color=color.magenta, axis=vector(0, 1, 0))
    side_arrow  = arrow(length=2,   shaftwidth=.1, color=color.orange,  axis=vector(0, 0, 1))

    us_pcb = box(
        pos=vector(0, 0, 0),
        length=0.025, height=0.4, width=1.1,
        color=vector(0.0, 0.4, 0.7)
    )
    us_eye_left = box(
        pos=vector(0.1, 0, -0.275),
        length=0.2, height=0.275, width=0.275,
        color=color.gray(0.75)
    )
    us_eye_right = box(
        pos=vector(0.1, 0, 0.275),
        length=0.2, height=0.275, width=0.275,
        color=color.gray(0.75)
    )
    bno_pcb = box(
        pos=vector(-CAM_BEHIND_SENSOR_CM * 0.025, CAM_ABOVE_SENSOR_CM * 0.025, 0),
        length=0.35, height=0.025, width=0.5,
        color=vector(0.1, 0.2, 0.6)
    )
    bno_chip = box(
        pos=vector(-CAM_BEHIND_SENSOR_CM * 0.025, (CAM_ABOVE_SENSOR_CM * 0.025) + 0.025, 0),
        length=0.1, height=0.025, width=0.1,
        color=color.black
    )
    servo_mount = box(
        pos=vector(-CAM_BEHIND_SENSOR_CM * 0.025, 0, 0),
        length=0.12, height=0.3, width=0.3,
        color=color.gray(0.15)
    )

    my_obj = compound([us_pcb, us_eye_left, us_eye_right, bno_pcb, bno_chip, servo_mount])

    target_point = sphere(
        pos=vector(0, 0, 0),
        radius=0.18,
        color=color.red,
        visible=False,
        make_trail=True,
        trail_type="curve",
        trail_color=color.red,
        trail_radius=0.06,
        retain=30
    )

    lbl_angles    = label(pos=vector(0, -5.2, 0), text="Roll: 0  Pitch: 0  Yaw: 0",          color=color.white,  height=12, box=False)
    lbl_dist      = label(pos=vector(0, -6.2, 0), text="Ultrasonic: --",                      color=color.cyan,   height=12, box=False)
    lbl_calib     = label(pos=vector(0, -7.2, 0), text="Calib: Sys=0 Gyr=0 Acc=0 Mg=0",      color=color.yellow, height=12, box=False)
    lbl_stereo    = label(pos=vector(0, -8.2, 0), text="Stereo Depth: --",                    color=color.green,  height=12, box=False)
    lbl_cartesian = label(
        pos=vector(0, -4.0, 0),
        text="Target Spatial Coordinates (XYZ): --",
        color=color.red,
        background=color.black,
        border=6,
        height=16,
        box=True,
        opacity=0.7,
    )

    SMOOTH_ALPHA = 0.15
    filtered_pos = vector(0, 0, 0)

    cv_thread = threading.Thread(target=opencv_thread, daemon=True)
    cv_thread.start()

    try:
        while True:
            rate(50)

            with shared["lock"]:
                if not shared["running"]:
                    break
                q0, q1, q2, q3        = shared["quat"]
                sys_c, gyr_c, acc_c, mg_c = shared["calib"]
                dist                  = shared["distance"]
                stereo_d              = shared["stereo_depth"]
                yaw_zero              = shared["yaw_zero"]
                pitch_zero            = shared["pitch_zero"]

            # ── Quaternion → Euler — UNCHANGED ───────────────────────────────
            offset_rad = math.radians(YAW_OFFSET_DEG)
            qo_w = math.cos(offset_rad / 2.0)
            qo_z = math.sin(offset_rad / 2.0)

            w_c = q0 * qo_w - q3 * qo_z
            x_c = q1 * qo_w + q2 * qo_z
            y_c = q2 * qo_w - q1 * qo_z
            z_c = q3 * qo_w + q0 * qo_z

            roll  = 0
            pitch = math.asin(max(-1.0, min(1.0, 2 * (w_c * y_c - z_c * x_c))))
            yaw   = math.atan2(2 * (w_c * z_c + x_c * y_c), 1 - 2 * (y_c * y_c + z_c * z_c))
            pitch -= pitch_zero
            yaw   -= yaw_zero

            with shared["lock"]:
                shared["current_pitch"] = pitch
                shared["current_yaw"]   = yaw

            # ── VPython orientation — UNCHANGED ──────────────────────────────
            k    = vector(math.cos(yaw) * math.cos(pitch), math.sin(pitch), math.sin(yaw) * math.cos(pitch))
            y_up = vector(0, 1, 0)
            s    = cross(k, y_up)
            v    = cross(s, k)
            vrot = v * math.cos(roll) + cross(k, v) * math.sin(roll)

            front_arrow.axis = k
            side_arrow.axis  = cross(k, vrot)
            up_arrow.axis    = vrot
            my_obj.axis      = k
            my_obj.up        = vrot

            front_arrow.length = 1.0
            side_arrow.length  = 1.0
            up_arrow.length    = 1.0

            # ── Target sphere — UNCHANGED, uses ultrasonic dist ───────────────
            if dist is not None and dist > 0:
                VIS_SCALE    = 0.05
                raw_pos      = k * dist * VIS_SCALE
                filtered_pos = filtered_pos + SMOOTH_ALPHA * (raw_pos - filtered_pos)
                target_point.pos     = filtered_pos
                target_point.visible = True
                lbl_cartesian.text   = (
                    f"Target (Sensor Origin) -> X: {k.x * dist:+.1f}cm  "
                    f"Y: {k.y * dist:+.1f}cm  Z: {k.z * dist:+.1f}cm"
                )
            else:
                target_point.visible = False
                target_point.clear_trail()
                filtered_pos       = vector(0, 0, 0)
                lbl_cartesian.text = "Target -> X: --  Y: --  Z: -- (Out of Sensor Range)"

            # ── Labels — UNCHANGED + new stereo depth label ───────────────────
            r_deg = math.degrees(roll)
            p_deg = math.degrees(pitch)
            y_deg = math.degrees(yaw)
            lbl_angles.text = f"Roll: {r_deg:+.1f}°  Pitch: {p_deg:+.1f}°  Yaw: {y_deg:+.1f}°"
            lbl_dist.text   = f"Ultrasonic: {dist:.1f} cm" if dist is not None else "Ultrasonic: --"
            lbl_calib.text  = f"Calib: Sys={sys_c} Gyr={gyr_c} Acc={acc_c} Mg={mg_c}"
            lbl_stereo.text = f"Stereo Depth: {stereo_d:.1f} cm" if stereo_d is not None else "Stereo Depth: --"

    except KeyboardInterrupt:
        print("\nShutting down cleanly...")
    finally:
        with shared["lock"]:
            shared["running"] = False


if __name__ == "__main__":
    main()
