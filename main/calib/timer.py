import numpy as np
import cv2 as cv
from picamera2 import Picamera2
import time

# ==============================
# CHESSBOARD SETTINGS
# ==============================

# Inner corners
chessboardSize = (11, 7)

# Square size in mm
squareSizeMm = (25.5 / 9) * 10

# ==============================
# TERMINATION CRITERIA
# ==============================

criteria = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==============================
# PREPARE OBJECT POINTS
# ==============================

objp = np.zeros(
    (chessboardSize[0] * chessboardSize[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:chessboardSize[0],
    0:chessboardSize[1]
].T.reshape(-1, 2)

objp *= squareSizeMm

# ==============================
# STORE POINTS
# ==============================

objPoints = []
imgPoints = []

# ==============================
# INITIALIZE PI CAMERA 2
# ==============================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1440, 1080)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Press SPACE to capture chessboard")
print("Press Q to finish and calibrate")

# ==============================
# CAPTURE LOOP
# ==============================

while True:

    # Capture frame
    frame = picam2.capture_array()

    # Convert RGB -> BGR for OpenCV
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboardSize,
        cv.CALIB_CB_ADAPTIVE_THRESH +
        cv.CALIB_CB_FAST_CHECK +
        cv.CALIB_CB_NORMALIZE_IMAGE
    )

    # Draw corners if found
    display = frame.copy()

    if ret:
        cv.drawChessboardCorners(
            display,
            chessboardSize,
            corners,
            ret
        )

    cv.imshow("Calibration", display)

    key = cv.waitKey(1)

    # SPACE = save frame
    if key == 32:

        if ret:

            objPoints.append(objp)

            corners2 = cv.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            imgPoints.append(corners2)

            print(f"Captured image {len(objPoints)}")

        else:
            print("Chessboard not detected")

    # Q = quit
    elif key == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================

cv.destroyAllWindows()
picam2.stop()

# ==============================
# CAMERA CALIBRATION
# ==============================

if len(objPoints) > 0:

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objPoints,
        imgPoints,
        gray.shape[::-1],
        None,
        None
    )

    print("\n==============================")
    print("CAMERA MATRIX")
    print("==============================")
    print(cameraMatrix)

    print("\n==============================")
    print("DISTORTION COEFFICIENTS")
    print("==============================")
    print(dist)

    print("\n==============================")
    print("RMS REPROJECTION ERROR")
    print("==============================")
    print(ret)

else:
    print("No valid chessboard captures.")