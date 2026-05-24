# from ultralytics import YOLO
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Load nano segmentation model
# model = YOLO("yolo11s-seg.pt")

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

kmeans = KMeans(n_clusters=3, n_init=10)
def kmeans_segmentation(img, k=3):
    x = img.reshape(-1, 3)
    kmeans.fit(x)
    seg_img = kmeans.cluster_centers_[kmeans.labels_]
    seg_img = seg_img.reshape(img.shape)
    return seg_img

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(
    n_clusters=15,
    batch_size=1000,
    n_init=3,
    random_state=42
)

def kmeans_segmentation_batch(img):
    x = img.reshape((-1, 3))


    labels = kmeans.fit_predict(x)

    seg = kmeans.cluster_centers_[labels]
    return seg.reshape(img.shape).astype("uint8")




import cv2
import numpy as np

def segment(frame, bbox=None, draw_border=True):
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
                m = cv2.resize(m, (x2 - x1, y2 - y1))
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], m)
            else:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
                mask = np.maximum(mask, m)

    annotated = frame.copy()

    # fill segmentation
    annotated[mask > 0] = (0, 255, 0)

    # optional border
    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

    return annotated, mask