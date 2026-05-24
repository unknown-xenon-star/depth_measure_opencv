import numpy as np
import cv2
from config import MASK_HSV

##############################################
#               Input
#                \/
#            Normalize
#                \/
#         [HSV RANGE MASKING]
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

def hsv_range_masking(img, blur_factor=7):
    # Blur image to reduce noise
    blur = cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)
    # Convert BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    lower_hsv = np.array(MASK_HSV[0])
    upper_hsv = np.array(MASK_HSV[1])
    
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask
    
# =====================================================
# AUTOMATIC GRABCUT
# =====================================================

def automatic_grabcut(
    img,
    mask,
    downscale_factor=2,
    iters=2
):
    h, w = img.shape[:2]
    # Downscale for performance optimization
    nh, nw = int(h / downscale_factor), int(w / downscale_factor)
    img_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Convert mask to grabcut format
    gc_mask = np.where(
        mask_small > 0,
        cv2.GC_PR_FGD,
        cv2.GC_BGD
    ).astype("uint8")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut refinement on downscaled image
    cv2.grabCut(
        img_small,
        gc_mask,
        None,
        bgdModel,
        fgdModel,
        iters,
        cv2.GC_INIT_WITH_MASK
    )

    # Final binary mask at small scale
    final_mask_small = np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Upscale back to original scale
    final_mask = cv2.resize(final_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return final_mask

def shape_refinement(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(refined, [largest], -1, 255, -1)
    return refined

def Morphological_Operations(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def main():
    
    # -----------------------------------------
    # Load image
    # -----------------------------------------
    image_path = "image.png"

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
    # HSV Range Masking
    # -----------------------------------------
    hsv_mask = hsv_range_masking(img)

    # -----------------------------------------
    # Automatic GrabCut
    # -----------------------------------------
    gmask = automatic_grabcut(
        img,
        hsv_mask,
        downscale_factor=2,
        iters=2
    )

    # -----------------------------------------
    # Shape Refinement
    # -----------------------------------------
    refined = shape_refinement(
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
    cv2.imwrite("hsv_mask.png", hsv_mask)
    cv2.imwrite("grabcut_mask.png", gmask)
    cv2.imwrite("final_mask.png", final_mask)
    cv2.imwrite("final_result.png", result)

    # -----------------------------------------
    # Show results
    # -----------------------------------------
    cv2.imshow("Original", img)
    cv2.imshow("HSV Mask", hsv_mask)
    cv2.imshow("GrabCut Mask", gmask)
    cv2.imshow("Final Mask", final_mask)
    cv2.imshow("Result", result)

    print("Processing Complete")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()