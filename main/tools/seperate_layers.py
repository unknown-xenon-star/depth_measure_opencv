import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import os

def save_images(imgs, output_dir="down", prefix="img"):
    """
    Save images with dtype and shape information in filename.

    Example filename:
    img_0_uint8_480x640x3.jpg
    """

    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(imgs):

        if img is None:
            print(f"[WARNING] Image {idx} is None. Skipping.")
            continue

        # Ensure numpy array
        img = np.asarray(img)
        
        dtype_name = img.dtype.name

        # Shape string
        shape_str = "x".join(map(str, img.shape))

        filename = (
            f"{prefix}_{idx}_{dtype_name}_{shape_str}.png"
        )

        path = os.path.join(output_dir, filename)

        # Handle float images safely
        save_img = img

        if img.dtype in [np.float32, np.float64]:

            finite_mask = np.isfinite(img)

            if np.any(finite_mask):

                min_val = np.min(img[finite_mask])
                max_val = np.max(img[finite_mask])

                if max_val > min_val:
                    save_img = cv2.normalize(
                        img,
                        None,
                        0,
                        255,
                        cv2.NORM_MINMAX
                    )

                save_img = np.uint8(save_img)

            else:
                save_img = np.zeros_like(img, dtype=np.uint8)
        cv2.imwrite(path, save_img)
        print(
            f"[SAVED] {path} | "
            f"dtype={img.dtype} | "
            f"shape={img.shape}"
        )

def save_new(img):
    h, w = img.shape[:2]

    # ----------------------------
    # 2. Convert to LAB color space
    # ----------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Optional (improves clustering for some images)
    lab = lab.astype(np.float32)

    # ----------------------------
    # 3. Reshape for K-means
    # ----------------------------
    Z = lab.reshape((-1, 3))
    Z = np.float32(Z)

    # ----------------------------
    # 4. K-means clustering
    # ----------------------------
    K = 4  # change to 5 or 6 if needed

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        Z,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # ----------------------------
    # 5. Reshape labels back to image
    # ----------------------------
    labels = labels.flatten()
    label_img = labels.reshape((h, w))

    # ----------------------------
    # 6. Create separate images per cluster
    # ----------------------------
    clusters = []

    for k in range(K):
        # binary mask for cluster k
        mask = (label_img == k).astype(np.uint8)

        # apply mask to original image
        cluster_img = cv2.bitwise_and(img, img, mask=mask)

        clusters.append(cluster_img)

    
    save_images(clusters)

def plot_rgb_planes(images, spacing=10):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    print("now")
    for i, img in enumerate(images):

        img = img[::5, ::5]   # reduce resolution 5x
        h, w, _ = img.shape

        # Normalize to 0–1
        img = img.astype(np.float32) / 255.0

        # Grid
        x = np.linspace(0, w, w)
        y = np.linspace(0, h, h)
        X, Y = np.meshgrid(x, y)

        # Z position for each image
        Z = np.full((h, w), i * spacing)

        print(i)
        # Plot as colored surface
        ax.plot_surface(
            X, Y, Z,
            facecolors=img,
            rstride=5,
            cstride=5,
            shade=False
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    print("RUN")
    plt.show()

def save_sepreated_layers(img, layered=True, cmap='viridis', n_layers=5):

    if layered:

        # Convert color image to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Create intensity ranges
        step = 256 // n_layers
        ranges = []

        for i in range(n_layers):
            low = i * step
            high = 255 if i == n_layers - 1 else (i + 1) * step
            ranges.append((low, high))

        parts = []

        for low, high in ranges:

            # Mask for intensity range
            mask = cv2.inRange(gray, low, high)

            # Extract region
            part = cv2.bitwise_and(gray, gray, mask=mask)

            parts.append(part)

        # Plot results
        fig, axes = plt.subplots(
            1,
            len(parts) + 1,
            figsize=(4 * (len(parts) + 1), 5)
        )

        # Original image
        axes[0].imshow(gray, cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Layered parts
        for i, part in enumerate(parts):
            axes[i + 1].imshow(part, cmap=cmap)
            axes[i + 1].set_title(
                f"{ranges[i][0]}-{ranges[i][1]}"
            )
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower_yellow = np.array([20, 100, 100])
        # upper_yellow = np.array([35, 255, 255])
        # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # lower_blue = np.array([90, 100, 100])
        # upper_blue = np.array([130, 255, 255])
        # mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # yellow_part = cv2.bitwise_and(img, img, mask=mask_yellow)
        # blue_part = cv2.bitwise_and(img, img, mask=mask_blue)
        print("BYE")
        # save([hsv, yellow_part, blue_part])
        save_new(hsv)