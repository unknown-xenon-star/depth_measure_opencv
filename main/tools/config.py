MODE = "static"

# =========================
# CONFIG
# =========================

# Camera field of view [degrees]
ALPHA               = 70        # horizontal FOV [degrees]
FOCAL_LENGTH        = 180.0     # pixels — tune to your camera
BASELINE            = 17.7      # camera baseline [cm]
DISPARITY_OFFSET    = 1.0       # avoids divide-by-zero at zero disparity
DISP_SCALE          = 0.5       # resize factor fed into SGBM (0.5 = quarter pixels)
MIN_AREA            = 50


LEFT_CAMERA_INDEX   = 0
RIGHT_CAMERA_INDEX  = 1
LEFT_IMAGE = "assest/im0.png"
RIGHT_IMAGE = "assest/im1.png"

MASK_HSV = [
    (173,114,153), (180,255,219)
]



