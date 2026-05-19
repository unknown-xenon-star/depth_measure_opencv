from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11s-seg.pt")

TARGET_CLASS = "person"


def detect_target(frame):
    results = model(frame)

    output = frame.copy()

    for r in results:

        boxes = r.boxes
        masks = r.masks

        if masks is None:
            continue

        for i, box in enumerate(boxes):

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name != TARGET_CLASS:
                continue

            conf = float(box.conf[0])

            # Get segmentation polygon
            polygon = masks.xy[i].astype(np.int32)

            # Random YOLO-like color
            color = (0, 255, 0)

            # Draw filled segmentation
            overlay = output.copy()
            cv2.fillPoly(overlay, [polygon], color)

            # Blend like YOLO visualization
            output = cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

            # Draw contour
            cv2.polylines(output, [polygon], True, color, 2)

            # Optional confidence text
            x, y = polygon[0]
            cv2.putText(
                output,
                f"{class_name} {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return output


cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    result_frame = detect_target(frame)

    cv2.imshow("FRAME", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()