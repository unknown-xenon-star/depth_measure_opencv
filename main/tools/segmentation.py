from ultralytics import YOLO
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Load nano segmentation model
# model = YOLO("yolov8n-seg.pt")

def kmeans_mask(img, k=3, target_cluster=0):

    # Flatten image
    pixels = img.reshape(-1, 3)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Select cluster
    mask = (labels == target_cluster)

    # Reshape to image
    mask = mask.reshape(img.shape[:2])

    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)

    return mask



def segment(frame, bbox=None):
    roi = frame
    result = model(roi, verbose=False)[0]
    if bbox:
        (x1, y1), (x2, y2) = bbox
        roi = frame[y1:y2, x1:x2]

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if result.masks is not None:

        for m in result.masks.data.cpu().numpy():

            m = (m > 0.5).astype(np.uint8) * 255

            if bbox:
                m = cv2.resize(m, (x2-x1, y2-y1))
                mask[y1:y2, x1:x2] = np.maximum(
                    mask[y1:y2, x1:x2],
                    m
                )
            else:
                mask = np.maximum(mask, m)

    annotated = frame.copy()
    annotated[mask > 0] = (0,255,0)

    return annotated, mask