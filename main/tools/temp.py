import numpy as np
from skimage import color, measure
from PIL import Image
import matplotlib.pyplot as plt


def kmeans_segment(image_path: str, max_k: int = 10, max_iter: int = 100, tol: float = 1e-4):
    """
    K-Means image segmentation in L*a*b* color space.
    Iterates K from 1 to max_k, stopping when all connected regions are obtained.

    Args:
        image_path : Path to the input image.
        max_k      : Maximum number of clusters (default 10).
        max_iter   : Max iterations per K-Means run.
        tol        : Convergence tolerance for centroid shift.

    Returns:
        best_labels : 2-D array of cluster indices (H x W).
        best_k      : The K value that produced the result.
        regions     : List of labelled connected-component regions.
    """
    # --- Load & convert to L*a*b* ---
    img_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb)                          # shape: (H, W, 3)
    H, W, _ = img_lab.shape
    pixels = img_lab.reshape(-1, 3)                           # (N, 3)

    best_labels = None
    best_k = 1

    # --- Outer loop: K = 1 … max_k ---
    for K in range(1, max_k + 1):

        # Initialize K centroids randomly from the pixel set
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(pixels), size=K, replace=False)
        centroids = pixels[idx].copy()                        # (K, 3)  → η₁…ηₙ

        labels = np.zeros(len(pixels), dtype=np.int32)

        # --- Repeat (inner K-Means loop) ---
        for _ in range(max_iter):

            # For each pixel, find index of closest centroid  → Cⁱ
            diffs = pixels[:, None, :] - centroids[None, :, :]   # (N, K, 3)
            distances = np.linalg.norm(diffs, axis=2)             # (N, K)
            new_labels = np.argmin(distances, axis=1)             # (N,)

            # Update each centroid to the mean of its assigned points  → ηₙ
            new_centroids = np.array([
                pixels[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centroids[k]
                for k in range(K)
            ])

            # Check convergence
            if np.linalg.norm(new_centroids - centroids) < tol:
                labels = new_labels
                break
            centroids = new_centroids
            labels = new_labels

        # Reshape labels back to image space
        label_image = labels.reshape(H, W)

        # Obtain the maximum number of connected segment regions
        connected = measure.label(label_image, connectivity=2)
        regions = measure.regionprops(connected)

        best_labels = label_image
        best_k = K

        # If all regions are obtained → display & break
        if len(regions) >= K:          # all cluster segments are connected
            print(f"All regions obtained at K = {K}  ({len(regions)} connected regions)")
            break
        else:
            print(f"K={K}: {len(regions)} regions found, incrementing K…")

    return best_labels, best_k, regions


# ── Display helper ────────────────────────────────────────────────────────────
def display_results(image_path: str, labels: np.ndarray, K: int, regions):
    img_rgb = np.array(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"L*a*b* K-Means Segmentation  (K = {K})", fontsize=14, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(labels, cmap="tab10")
    axes[1].set_title(f"Cluster Labels  (K={K})")
    axes[1].axis("off")

    # Colour each cluster with its mean L*a*b* → RGB colour
    H, W = labels.shape
    img_lab = color.rgb2lab(img_rgb.astype(np.float32) / 255.0)
    seg_lab = np.zeros_like(img_lab)
    for k in range(K):
        mask = labels == k
        if mask.any():
            seg_lab[mask] = img_lab[mask].mean(axis=0)
    seg_rgb = np.clip(color.lab2rgb(seg_lab), 0, 1)

    axes[2].imshow(seg_rgb)
    axes[2].set_title("Segmented Image")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Result saved → segmentation_result.png")
    print(f"Total connected regions: {len(regions)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = "input.jpg"
    labels, K, regions = kmeans_segment(path, max_k=10)
    display_results(path, labels, K, regions)