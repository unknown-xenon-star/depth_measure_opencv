"""
hybrid_segmentation.py
───────────────────────────────────────────────────────────────────────────────
Hybrid Object Segmentation combining:
  1. RGB Image Normalization
  2. L*a*b* Colour Space Conversion
  3. Adaptive K-Means Segmentation  (K=1..10, SSE-driven)
  4. Automatic GrabCut Segmentation (Orchard-Bouman GMM init, no user input)
  5. Shape Refinement               (energy-based occlusion removal)
  6. Morphological Operations       (hole-fill + Canny edge refinement)

Based on:
  "Object Segmentation Based on the Integration of Adaptive K-means
   and GrabCut Algorithm" — Prabu S. & Joseph Abraham Sundar K., WiSPNET 2022

Drop-in replacement for add_HSV_filter() in the main stereo pipeline.
Usage:
    from tools.hybrid_segmentation import hybrid_segment
    mask = hybrid_segment(frame, hsv_hint=None)
"""

import cv2
import numpy as np
# from sklearn.mixture import GaussianMixture


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — RGB Image Normalization
# ═══════════════════════════════════════════════════════════════════════════════
def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    """
    Eliminate light variations and bright/shaded regions.
    Each channel divided by pixel-wise sum of all channels, scaled to [0,255].
    Equations (1)–(6) from the paper.
    """
    img_f = image.astype(np.float32) + 1e-6          # avoid zero-division
    channel_sum = img_f.sum(axis=2, keepdims=True)
    normalized = (img_f / channel_sum) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Colour Space Conversion  RGB → L*a*b*
# ═══════════════════════════════════════════════════════════════════════════════
def _to_lab(image: np.ndarray) -> np.ndarray:
    """BGR → L*a*b* via OpenCV (internally does XYZ intermediate step)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Adaptive K-Means Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
def _adaptive_kmeans(lab_image: np.ndarray,
                     k_min: int = 2,
                     k_max: int = 10) -> np.ndarray:
    """
    Cluster the L*a*b* image pixels with K-Means.
    K is chosen adaptively (1–10) using the SSE elbow criterion.
    Returns a binary mask where the LARGEST connected foreground cluster = 255.
    """
    h, w = lab_image.shape[:2]
    pixels = lab_image.reshape(-1, 3).astype(np.float32)

    best_k     = k_min
    best_sse   = np.inf
    prev_sse   = np.inf
    sse_values = []

    # ── Elbow search ──────────────────────────────────────────────────────────
    for k in range(k_min, k_max + 1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria,
            attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )
        sse = float(np.sum((pixels - centers[labels.flatten()]) ** 2))
        sse_values.append(sse)

        # Elbow: stop when improvement falls below 15 %
        if prev_sse < np.inf:
            improvement = (prev_sse - sse) / (prev_sse + 1e-9)
            if improvement < 0.15:
                best_k = k - 1
                break
        prev_sse = sse
        best_k   = k

    # ── Re-run with chosen K ───────────────────────────────────────────────────
    best_k = max(best_k, k_min)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, centers = cv2.kmeans(
        pixels, best_k, None, criteria,
        attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )

    label_map = labels.reshape(h, w)

    # ── Pick foreground cluster: highest L* mean (brightest object) ─────────
    # Alternatively you could pick smallest/largest area; brightest works well
    # for coloured objects on a neutral background.
    best_cluster   = 0
    best_area      = 0
    for k in range(best_k):
        cluster_mask = (label_map == k).astype(np.uint8)
        # count connected components — pick cluster with most connected regions
        n_comp, comp_labels = cv2.connectedComponents(cluster_mask)
        area = np.sum(cluster_mask)
        if area > best_area:
            best_area    = area
            best_cluster = k

    # Use the cluster with the highest average 'a' channel (reddish/green hue)
    # to target salient coloured foreground objects
    cluster_scores = []
    for k in range(best_k):
        mask_k  = (label_map == k)
        lab_px  = lab_image[mask_k].astype(float)
        # score = chroma magnitude in a*b* plane (salient coloured objects score high)
        chroma  = np.mean(np.sqrt((lab_px[:, 1] - 128) ** 2 +
                                  (lab_px[:, 2] - 128) ** 2)) if len(lab_px) else 0
        cluster_scores.append(chroma)

    best_cluster = int(np.argmax(cluster_scores))
    kmeans_mask  = ((label_map == best_cluster) * 255).astype(np.uint8)

    return kmeans_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Automatic GrabCut Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
def _auto_grabcut(bgr_image: np.ndarray,
                  hint_mask: np.ndarray) -> np.ndarray:
    """
    Automatic GrabCut — no user rectangle required.

    Initialisation follows the paper (Table II):
      1. Entire image = unknown region
      2. Orchard-Bouman clustering (mimicked via sklearn GMM K=2) splits
         image into FG / BG
      3. Two full-covariance GMMs (K=5) built from those pixel sets
      4-7. GrabCut iterates to convergence

    hint_mask: binary uint8 (255 = likely foreground) from K-Means step,
               used to seed the initial FG/BG split.
    """
    h, w = bgr_image.shape[:2]

    # ── Build trimap from K-Means mask (Step 2 of Table II) ─────────────────
    # Erode the hint mask to get a confident FG seed
    kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    sure_fg = cv2.erode(hint_mask,  kernel_fg, iterations=1)
    sure_bg = cv2.dilate(hint_mask, kernel_bg, iterations=3)
    sure_bg = cv2.bitwise_not(sure_bg)           # invert → pixels far from object

    # GrabCut mask convention:
    #   0 = cv2.GC_BGD  (definite BG)
    #   1 = cv2.GC_FGD  (definite FG)
    #   2 = cv2.GC_PR_BGD (probable BG)
    #   3 = cv2.GC_PR_FGD (probable FG)
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)   # start: probable BG
    gc_mask[sure_fg == 255] = cv2.GC_FGD                        # certain FG
    gc_mask[sure_bg == 255] = cv2.GC_BGD                        # certain BG

    # If no certain FG exists, fall back to centre-based rectangle
    if np.sum(sure_fg) == 0:
        margin  = max(10, min(h, w) // 8)
        rect    = (margin, margin, w - 2 * margin, h - 2 * margin)
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr_image, gc_mask, rect,
                    bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    else:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr_image, gc_mask, None,
                    bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Pixels labelled FGD or PR_FGD → foreground
    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    return fg_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Shape Refinement  (Eq. 12 from paper)
# ═══════════════════════════════════════════════════════════════════════════════
def _shape_refinement(mask: np.ndarray,
                      lab_image: np.ndarray,
                      mu: float = 0.5) -> np.ndarray:
    """
    Energy-based shape refinement:
        E = OEi − µ * Et_s
    OEi = gradient magnitude of L channel (original image energy)
    Et_s = distance transform of current mask (shape energy term)
    Pixels where E > 0 are retained as foreground.
    """
    L_channel = lab_image[:, :, 0].astype(np.float32)

    # Original image energy: Sobel gradient magnitude on L channel
    gx   = cv2.Sobel(L_channel, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(L_channel, cv2.CV_32F, 0, 1, ksize=3)
    OEi  = cv2.magnitude(gx, gy)
    OEi  = cv2.normalize(OEi, None, 0, 1, cv2.NORM_MINMAX)

    # Shape energy: distance transform of current mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    Et_s = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

    # Energy expression
    E = OEi - mu * Et_s

    # Retain mask pixels where original energy dominates
    refined = mask.copy()
    suppress = (E < 0) & (mask == 255)
    refined[suppress] = 0

    return refined


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Morphological Operations
# ═══════════════════════════════════════════════════════════════════════════════
def _morphological_ops(mask: np.ndarray) -> np.ndarray:
    """
    1. Close small gaps
    2. Partial filling (flood-fill black holes inside connected white regions)
    3. Remove small isolated blobs
    4. Canny-guided edge refinement
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Close → fill thin gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # Open  → remove isolated specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)

    # ── Hole filling (partial fill via flood-fill on inverted mask) ──────────
    h, w     = mask.shape
    filled   = mask.copy()
    flood_in = cv2.bitwise_not(filled)
    seed_pt  = (0, 0)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_in, flood_mask, seed_pt, 255)
    holes    = cv2.bitwise_not(flood_in)
    filled   = cv2.bitwise_or(filled, holes)

    # ── Keep only the largest connected component ─────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        filled  = (labels == largest).astype(np.uint8) * 255

    return filled


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def hybrid_segment(bgr_frame: np.ndarray,
                   hsv_hint: np.ndarray | None = None,
                   use_grabcut: bool = True) -> np.ndarray:
    """
    Full hybrid segmentation pipeline (6 steps from the paper).

    Parameters
    ----------
    bgr_frame  : H×W×3 BGR image from camera
    hsv_hint   : optional H×W uint8 binary mask from HSV filter —
                 when provided, it is combined (AND) with K-Means result
                 to restrict the foreground region further.
    use_grabcut: set False for low-power mode (skips GrabCut, ~3× faster)

    Returns
    -------
    mask : H×W uint8, 255 = foreground object, 0 = background
    """
    # ── Step 1 — Normalise ────────────────────────────────────────────────────
    norm_rgb = _normalize_rgb(bgr_frame)

    # ── Step 2 — L*a*b* conversion ───────────────────────────────────────────
    lab = _to_lab(norm_rgb)

    # ── Step 3 — Adaptive K-Means ────────────────────────────────────────────
    kmeans_mask = _adaptive_kmeans(lab, k_min=2, k_max=8)

    # Optionally combine with HSV hint for colour-constrained tracking
    if hsv_hint is not None:
        hsv_bin     = (hsv_hint > 0).astype(np.uint8) * 255
        kmeans_mask = cv2.bitwise_and(kmeans_mask, hsv_bin)

    # Early-exit if K-Means found nothing
    if np.sum(kmeans_mask) == 0:
        return np.zeros(bgr_frame.shape[:2], dtype=np.uint8)

    # ── Step 4 — Automatic GrabCut ───────────────────────────────────────────
    if use_grabcut:
        gc_mask = _auto_grabcut(bgr_frame, kmeans_mask)
        # Intersect with K-Means result for best of both worlds
        combined = cv2.bitwise_and(kmeans_mask, gc_mask)
        # If GrabCut wiped everything out, fall back to K-Means only
        if np.sum(combined) < 100:
            combined = kmeans_mask
    else:
        combined = kmeans_mask

    # ── Step 5 — Shape Refinement ────────────────────────────────────────────
    refined = _shape_refinement(combined, lab, mu=0.5)

    # ── Step 6 — Morphological Operations ───────────────────────────────────
    final_mask = _morphological_ops(refined)

    return final_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Lightweight alias — same interface as add_HSV_filter
# ═══════════════════════════════════════════════════════════════════════════════
def hybrid_mask(bgr_frame: np.ndarray,
                blur_k: int = 7,
                hsv_range: list | None = None) -> np.ndarray:
    """
    Drop-in replacement for add_HSV_filter(frame, blur_k, hsv_range).

    Runs the full hybrid pipeline; if hsv_range is given the HSV mask
    is generated internally and fed as a hint to the K-Means step.
    """
    hsv_hint = None
    if hsv_range is not None and len(hsv_range) == 2:
        blurred  = cv2.GaussianBlur(bgr_frame, (blur_k, blur_k), 0)
        hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lo       = np.array(hsv_range[0], dtype=np.uint8)
        hi       = np.array(hsv_range[1], dtype=np.uint8)
        hsv_hint = cv2.inRange(hsv, lo, hi)

    return hybrid_segment(bgr_frame, hsv_hint=hsv_hint, use_grabcut=True)
