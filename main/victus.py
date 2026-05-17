import cv2
LEFT_CAMERA_INDEX = 0

def run_live():

    cap_left = cv2.VideoCapture(LEFT_CAMERA_INDEX)
    cap_right = cv2.VideoCapture(RIGHT_CAMERA_INDEX)

    if not cap_left.isOpened():
        raise RuntimeError("Could not open left camera")

    if not cap_right.isOpened():
        raise RuntimeError("Could not open right camera")

    while True:

        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Frame capture failed")
            break

        disparity_map, depth = compute_depth(frame_l, frame_r)

        cv2.imshow("Left Camera", frame_l)
        cv2.imshow("Right Camera", frame_r)
        cv2.imshow("Disparity / Depth Map", disparity_map)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_left.release()
    # cap_right.release()
    cv2.destroyAllWindows()

run_live()