import cv2
import numpy as np


def resize(img, factor):
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)

def masked_percentile_depth(
    depth_map:  np.ndarray,
    mask:       np.ndarray,
    bbox,
    percentile: float = 50,
) -> float | None:
    """
    Median (or any percentile) depth inside the object bbox,
    restricted to pixels that are masked, finite, and positive.
    """
    (x1, y1), (x2, y2) = bbox
    values = depth_map[y1:y2, x1:x2][mask[y1:y2, x1:x2] > 0]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, percentile)) if values.size > 0 else None