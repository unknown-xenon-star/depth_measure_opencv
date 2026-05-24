from sklearn.cluster import KMeans
import numpy as np
import cv2

##############################################
#               Input
#                \/
#            Normalize
#                \/
#      Color CONVERSION TO LAB
#                \/
#         [ADAPTIVE GKMEANS]
#                \/
#          AUTOMATIC GRABCUT
#                \/
#         SHAPE REFINEMENT
#                \/
#     MORPHOLOGICAL OPERATIONS
#                \/
#              Results
##############################################

def normaliztion(img):
    img = img.astype(np.float32)
    s = img.sum(axis=2, keepdims=True) + 1e-6
    norm = (img / s) * 255 
    return norm.astype(np.uint8)

def color_conversion(img, BGR: bool = True):
    if BGR:
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def adaptive_kmeans( img, k=4 ):
    pixels = img.reshape((-1, 3))
    kmeans = KMeans( n_clusters=k, n_init=10 )
    labels = kmeans.fit_predict(pixels)
    seg = kmeans.cluster_centers_[labels]
    seg = seg.reshape(img.shape)
    seg = seg.astype(np.uint8)
    # Largest cluster
    cluster_id = np.argmax( np.bincount(labels))
    mask = (labels == cluster_id)
    mask = mask.reshape( img.shape[:2])
    mask = ( mask.astype(np.uint8) * 255)
    return seg, mask
    
# =====================================================
# AUTOMATIC GRABCUT
# =====================================================

def automatic_grabcut(
    img,
    mask
):

    # Convert mask to grabcut format
    gc_mask = np.where(
        mask > 0,
        cv2.GC_PR_FGD,
        cv2.GC_BGD
    ).astype("uint8")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut refinement
    cv2.grabCut(
        img,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    # Final binary mask
    final_mask = np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return final_mask

def shape_refShape_Refinement(mask):
    contours, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(mask)
    if contours:
        largest = max( contours, key=cv2.contourArea )
        cv2.drawContours( refined, [largest], -1, 255, -1)
    return refined

def Morphological_Operations(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    return mask

def canny(img):
    cv2.imshow("canny", cv2.Canny(img, 125, 175))

def main():
    
    # -----------------------------------------
    # Load image
    # -----------------------------------------
    image_path = "screen.png"

    img = cv2.imread(image_path)

    if img is None:
        print("ERROR: Image not found")
        return

    # -----------------------------------------
    # Resize (optional)
    # -----------------------------------------
    scale = 0.7

    img = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )

    # -----------------------------------------
    # Normalize
    # -----------------------------------------
    norm = normaliztion(img)

    # -----------------------------------------
    # Convert to LAB
    # -----------------------------------------
    lab = color_conversion(norm)

    # -----------------------------------------
    # Adaptive KMeans
    # -----------------------------------------
    seg, kmask = adaptive_kmeans(
        lab,
        k=4
    )

    # -----------------------------------------
    # Automatic GrabCut
    # -----------------------------------------
    gmask = automatic_grabcut(
        img,
        kmask
    )

    # -----------------------------------------
    # Shape Refinement
    # -----------------------------------------
    refined = shape_refShape_Refinement(
        gmask
    )

    # -----------------------------------------
    # Morphological Operations
    # -----------------------------------------
    final_mask = Morphological_Operations(
        refined
    )

    # -----------------------------------------
    # Apply mask
    # -----------------------------------------
    result = cv2.bitwise_and(
        img,
        img,
        mask=final_mask
    )

    # -----------------------------------------
    # Save results
    # -----------------------------------------
    cv2.imwrite("kmeans_segment.png", seg)
    cv2.imwrite("grabcut_mask.png", gmask)
    cv2.imwrite("final_mask.png", final_mask)
    cv2.imwrite("final_result.png", result)

    # -----------------------------------------
    # Show results
    # -----------------------------------------
    cv2.imshow("Original", img)
    cv2.imshow("KMeans Mask", kmask)
    cv2.imshow("GrabCut Mask", gmask)
    cv2.imshow("Final Mask", final_mask)
    cv2.imshow("Result", result)

    print("Processing Complete")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()