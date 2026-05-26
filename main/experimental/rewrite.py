import math
import os
import threading
import time

import cv2
import numpy as np
import serial
from picamera2 import Picamera2
# ── Config ──────────────────────────────────────────────────────────────────
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

ANGLE_MIN    = 0
ANGLE_MAX    = 180
SERVO_CENTER = 90
PAN_OFFSET   = 0
TILT_OFFSET  = 10
TILT_HOME    = 90
PAN_HOME     = 90

YAW_OFFSET_DEG = -45.0

# ── Sensor-Camera Spatial Offset ────────────────────────────────────────────
CAM_BEHIND_SENSOR_CM  = 10.0   
CAM_ABOVE_SENSOR_CM   =  7.0   


def clamp_angle(value):
    return max(ANGLE_MIN, min(ANGLE_MAX, int(round(value))))

def to_servo_angles(horizontal_angle, vertical_angle):
    pan  = clamp_angle(PAN_HOME  + horizontal_angle + PAN_OFFSET)
    tilt = clamp_angle(TILT_HOME + vertical_angle   + TILT_OFFSET)
    return pan, tilt
