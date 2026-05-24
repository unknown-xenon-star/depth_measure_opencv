import cv2
import numpy as np


def auto_canny(image, sigma=0.33):
    """Automatically compute Canny thresholds using median of pixel intensities."""
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper), lower, upper


def adaptive_blur(gray):
    """Choose blur kernel size based on image resolution."""
    h, w = gray.shape
    area = h * w
    if area < 200_000:
        ksize = (3, 3)
    elif area < 1_000_000:
        ksize = (5, 5)
    else:
        ksize = (7, 7)
    return cv2.GaussianBlur(gray, ksize, 0), ksize


def select_best_contours(contours, img_area, min_ratio=0.001, max_ratio=0.95):
    """Filter contours by relative area to ignore noise and full-image artifacts."""
    return [
        c for c in contours
        if min_ratio < cv2.contourArea(c) / img_area < max_ratio
    ]


def fill_holes(mask):
    """Flood-fill from border to close internal holes in the mask."""
    flood = mask.copy()
    h, w = flood.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 255)
    inverted = cv2.bitwise_not(flood)
    return mask | inverted


def segment(image_path: str, sigma: float = 0.33, top_n: int = 3):
    """
    Automatic adaptive edge-based segmentation pipeline.

    Parameters
    ----------
    image_path : str   Path to input image.
    sigma      : float Controls Canny threshold spread (lower → more edges).
    top_n      : int   How many largest contours to include in the mask.

    Returns
    -------
    dict with keys: original, gray, blur, edges, mask, result, meta
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open '{image_path}'")

    # ── Auto-resize: cap longest side at 1200 px ──────────────────────────────
    h, w = img.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    img_area = img.shape[0] * img.shape[1]

    # ── Grayscale ─────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Adaptive blur ─────────────────────────────────────────────────────────
    blur, ksize = adaptive_blur(gray)

    # ── Automatic Canny ───────────────────────────────────────────────────────
    edges, t_low, t_high = auto_canny(blur, sigma=sigma)

    # ── Morphological close: bridge small gaps in edges ───────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ── Contours ──────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ── Filter + rank ─────────────────────────────────────────────────────────
    good = select_best_contours(contours, img_area)
    good_sorted = sorted(good, key=cv2.contourArea, reverse=True)
    selected = good_sorted[:top_n]

    # ── Build mask ────────────────────────────────────────────────────────────
    mask = np.zeros(gray.shape, np.uint8)
    if selected:
        cv2.drawContours(mask, selected, -1, 255, thickness=cv2.FILLED)

    # Fill interior holes
    mask = fill_holes(mask)

    # Light smoothing on mask boundary
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── Apply mask ────────────────────────────────────────────────────────────
    result = cv2.bitwise_and(img, img, mask=mask)

    # White background for cleaner look
    bg = np.full_like(img, 255)
    bg_mask = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(bg, bg, mask=bg_mask)
    result_white_bg = cv2.add(result, bg_part)

    meta = {
        "original_shape": (h, w),
        "resized_shape": img.shape[:2],
        "scale": scale,
        "blur_kernel": ksize,
        "canny_low": t_low,
        "canny_high": t_high,
        "contours_found": len(contours),
        "contours_used": len(selected),
    }

    return {
        "original": img,
        "gray": gray,
        "blur": blur,
        "edges": edges_closed,
        "mask": mask,
        "result": result_white_bg,
        "meta": meta,
    }


# ── CLI / demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Automatic Adaptive Edge Segmentation")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--sigma", type=float, default=0.33,
                    help="Canny sigma (default 0.33)")
    ap.add_argument("--top-n", type=int, default=3,
                    help="Number of largest contours to keep (default 3)")
    ap.add_argument("--save", action="store_true",
                    help="Save outputs instead of displaying them")
    args = ap.parse_args()

    try:
        out = segment(args.image, sigma=args.sigma, top_n=args.top_n)
    except FileNotFoundError as e:
        print(e); sys.exit(1)

    m = out["meta"]
    print(f"\n── Segmentation Report ──────────────────────────")
    print(f"  Original size  : {m['original_shape'][1]}×{m['original_shape'][0]}")
    print(f"  Processed size : {m['resized_shape'][1]}×{m['resized_shape'][0]}"
          f"  (scale {m['scale']:.2f})")
    print(f"  Blur kernel    : {m['blur_kernel']}")
    print(f"  Canny thresholds: {m['canny_low']} / {m['canny_high']}  (auto)")
    print(f"  Contours found : {m['contours_found']}")
    print(f"  Contours used  : {m['contours_used']}")
    print(f"─────────────────────────────────────────────────\n")

    if args.save:
        cv2.imwrite("out_edges.jpg",  out["edges"])
        cv2.imwrite("out_mask.jpg",   out["mask"])
        cv2.imwrite("out_result.jpg", out["result"])
        print("Saved: out_edges.jpg  out_mask.jpg  out_result.jpg")
    else:
        panels = {
            "Original":  out["original"],
            "Edges":     out["edges"],
            "Mask":      out["mask"],
            "Segmented": out["result"],
        }
        for title, frame in panels.items():
            cv2.imshow(title, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()