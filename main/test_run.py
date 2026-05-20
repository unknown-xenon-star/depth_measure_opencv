
import cv2

# ── Main loop ──────────────────────────────────────────────────────────────────

# cap = cv2.VideoCapture("assest/video.mp4")
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    if not ret1:
        break

    cv2.imshow("RIGHT CAMERA", frame_right)
    cv2.imshow("LEFT CAMERA",  frame_left)

    if cv2.pollKey() & 0xFF == ord('q'):   # non-blocking — saves ~1 ms/frame
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()