import cv2
import numpy as np


def resize(img, factor):
    return cv2.resize(img, (0, 0), fx=factor, fy=factor)

def masked_percentile_depth(
    depth_map:  np.ndarray,
    mask:       np.ndarray,
    bbox,
    percentile: float = 50,
    near_clip: float = 5,
    far_clip: float = 95,
    min_depth: float = 0.0,
) -> float | None:
    """
    Median (or any percentile) depth inside the object bbox,
    restricted to pixels that are masked, finite, and positive.
    """
    (x1, y1), (x2, y2) = bbox
    values = depth_map[y1:y2, x1:x2][mask[y1:y2, x1:x2] > 0]
    values = values[np.isfinite(values) & (values > min_depth)]
    if values.size == 0:
        return None
    near = np.percentile(values, near_clip)
    far  = np.percentile(values, far_clip)
    values = values[ (values >= near) & (values <= far) ]
    return float(np.percentile(values, percentile)) if values.size > 0 else None

def annotate(frame: np.ndarray, tracking: bool, depth) -> None:
    """Draw tracking status and depth label onto *frame* in-place."""
    if tracking:
        label = f"Distance: {round(depth, 2) if depth is not None else 'NaN'} cm"
        cv2.putText(frame, "TRACKING", (75,  50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame, label,       (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
    else:
        cv2.putText(frame, "TRACKING LOST", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
