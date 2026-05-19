import cv2
import numpy as np

cap = cv2.VideoCapture(0)
n_frame = None
while True:
    okTrue, frame = cap.read()

    if not okTrue:
        break

    cv2.imshow("FRAME",frame)
    n_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
