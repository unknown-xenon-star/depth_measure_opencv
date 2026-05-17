import cv2


# =========================
# CONFIG
# =========================

MODE = "static"          # "static" or "live"

LEFT_IMAGE = "im0.png"
RIGHT_IMAGE = "im1.png"

LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = 1


# =========================
# STATIC IMAGE MODE
# =========================

def load_static_images(left_path, right_path):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)

    if left is None or right is None:
        raise FileNotFoundError("Could not load one or both images.")

    cv2.imshow("Left Image", cv2.resize(left, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow("Right Image", cv2.resize(right, (0, 0), fx=0.5, fy=0.5))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# LIVE CAMERA MODE
# =========================

def live_stereo_capture(cam_left, cam_right):
    cap_left = cv2.VideoCapture(cam_left)
    cap_right = cv2.VideoCapture(cam_right)

    if not cap_left.isOpened():
        raise RuntimeError(f"Could not open left camera: {cam_left}")

    if not cap_right.isOpened():
        raise RuntimeError(f"Could not open right camera: {cam_right}")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Failed to capture frames.")
            break

        cv2.imshow("Left Camera", frame_l)
        cv2.imshow("Right Camera", frame_r)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if MODE == "static":
        load_static_images(LEFT_IMAGE, RIGHT_IMAGE)

    elif MODE == "live":
        live_stereo_capture(
            LEFT_CAMERA_INDEX,
            RIGHT_CAMERA_INDEX
        )

    else:
        print("Invalid MODE")