import numpy as np
import cv2 as cv
import glob

# ==============================
# CHESSBOARD SETTINGS
# ==============================

# CHANGED: Updated to match the actual inner corners of your board (11x7)
chessboardSize = (11, 7)

# CHANGED: Added your exact square size in millimeters (25.5 cm / 9 squares)
squareSizeMm = 25.5 / 9 * 10  # Roughly 28.3333 mm

# Image resolution
frameSize = (1440, 1080)

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

# Create rows and columns filled with zeros based on new board size
objp = np.zeros(
    (chessboardSize[0] * chessboardSize[1], 3),
    np.float32
)

# Fill x and y coordinates
objp[:, :2] = np.mgrid[
    0:chessboardSize[0],
    0:chessboardSize[1]
].T.reshape(-1, 2)

# CHANGED: Scale coordinates by the real-world square size
objp *= squareSizeMm

# ==============================
# STORE POINTS
# ==============================

# 3D real-world points
objPoints = []

# 2D image points
imgPoints = []

# ==============================
# LOAD IMAGES
# ==============================

# CHANGED: Modified extension to match your '.jpg' files
images = glob.glob('*.jpg')

# ==============================
# PROCESS EACH IMAGE
# ==============================

for image in images:

    print("Processing:", image)

    # Read image
    img = cv.imread(image)
    if img is None:
        print(f"Warning: Could not read {image}")
        continue

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(
        gray,
        chessboardSize,
        None
    )

    # If corners found
    if ret == True:

        # Save real-world points
        objPoints.append(objp)

        # Refine corner accuracy
        corners2 = cv.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Save image points
        imgPoints.append(corners2)

        # Draw corners
        cv.drawChessboardCorners(
            img,
            chessboardSize,
            corners2,
            ret
        )

        # Show image
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print(f"Failed to find corners in {image}")

# Close all windows
cv.destroyAllWindows()

# Ensure we actually found corners before running calibration
if len(objPoints) > 0:
    # ==============================
    # CAMERA CALIBRATION
    # ==============================

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objPoints,
        imgPoints,
        gray.shape[::-1], # Best practice: Uses actual image dimension dynamically
        None,
        None
    )

    # ==============================
    # PRINT RESULTS
    # ==============================

    print("\nCAMERA MATRIX:")
    print(cameraMatrix)

    print("\nDISTORTION COEFFICIENTS:")
    print(dist)

    print("\nROTATION VECTORS:")
    print(rvecs)

    print("\nTRANSLATION VECTORS (in mm):")
    print(tvecs)
else:
    print("Error: No chessboard corners were detected in any images.")
