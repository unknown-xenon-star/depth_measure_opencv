import cv2
import numpy as np
import glob

# ============================================
# CHESSBOARD SETTINGS
# ============================================

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 2.5   # cm

# ============================================
# LEFT CAMERA INTRINSICS
# ============================================

K_left = np.array([
    [755.79240334, 0.0, 336.47450076],
    [0.0, 755.36361434, 255.72185014],
    [0.0, 0.0, 1.0]
])

dist_left = np.array([
    [-5.02451202e-01,
      7.33283229e-01,
     -4.02834197e-03,
      8.96470971e-06,
     -1.55985928e+00]
])

# ============================================
# RIGHT CAMERA INTRINSICS
# ============================================

K_right = np.array([
    [734.90387831, 0.0, 330.55013381],
    [0.0, 735.03555766, 236.51655862],
    [0.0, 0.0, 1.0]
])

dist_right = np.array([
    [-0.46708432,
      0.5110633,
     -0.00213905,
     -0.00105646,
     -0.80638573]
])

# ============================================
# PREPARE REAL-WORLD CHESSBOARD POINTS
# ============================================

objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ============================================
# STORAGE ARRAYS
# ============================================

objpoints = []

imgpoints_left = []
imgpoints_right = []

# ============================================
# LOAD IMAGE PAIRS
# ============================================

left_images = sorted(glob.glob("LEFT/*.jpg"))
right_images = sorted(glob.glob("RIGHT/*.jpg"))

# ============================================
# FIND CHESSBOARD CORNERS
# ============================================

for left_path, right_path in zip(left_images, right_images):

    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        print("Error loading image pair")
        continue

    gray_left = cv2.cvtColor(
        img_left,
        cv2.COLOR_BGR2GRAY
    )

    gray_right = cv2.cvtColor(
        img_right,
        cv2.COLOR_BGR2GRAY
    )

    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left,
        CHESSBOARD_SIZE
    )

    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right,
        CHESSBOARD_SIZE
    )

    # If corners found in BOTH images
    if ret_left and ret_right:

        objpoints.append(objp)

        # Improve corner accuracy
        corners_left = cv2.cornerSubPix(
            gray_left,
            corners_left,
            (11,11),
            (-1,-1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        corners_right = cv2.cornerSubPix(
            gray_right,
            corners_right,
            (11,11),
            (-1,-1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        print(f"Corners detected: {left_path}")

    else:
        print(f"Failed: {left_path}")

# ============================================
# STEREO CALIBRATION (EXTRINSIC)
# ============================================

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K_left,
    dist_left,
    K_right,
    dist_right,
    gray_left.shape[::-1],

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-5
    ),

    flags = cv2.CALIB_FIX_INTRINSIC
)

# ============================================
# PRINT RESULTS
# ============================================

print("\n================================")
print("STEREO CALIBRATION COMPLETE")
print("================================")

print("\nRotation Matrix (R):")
print(R)

print("\nTranslation Matrix (T):")
print(T)

# ============================================
# BASELINE CALCULATION
# ============================================

baseline = np.linalg.norm(T)

print(f"\nCalculated Baseline: {baseline:.2f} cm")

# ============================================
# COMPARE WITH EXPECTED BASELINE
# ============================================

expected_baseline = 5.0  # cm

error = abs(baseline - expected_baseline)

print(f"Expected Baseline: {expected_baseline} cm")
print(f"Baseline Error: {error:.2f} cm")