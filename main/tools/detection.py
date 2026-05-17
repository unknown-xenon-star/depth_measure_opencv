import cv2
import numpy as np

def crop_cord(x,y,w,h, depth_map):
    cx = x + w//4
    cy = y + h//4

    cw = w//2
    ch = h//2

    center_depth = depth_map[cy:cy+ch, cx:cx+cw]
    object_depth = depth_map[y:y+h, x:x+w]


    # Remove invalid values
    # valid = object_depth[np.isfinite(object_depth)]
    valid = object_depth[np.isfinite(object_depth)]
    valid = valid[valid > 0]

    if len(valid) > 0:
        avg_depth = np.mean(valid)
        median_depth = np.median(valid)

        print("Average depth:", avg_depth)
        print("Median depth:", median_depth)
    else:
        print("No valid depth values found")



def segmentation(depth_map, contour_mask):
    mask = np.zeros(depth_map.shape, dtype=np.uint8)

    cv2.drawContours(mask, [contour_mask], -1, 255, -1)

    object_pixels = depth_map[mask == 255]

    valid = object_pixels[np.isfinite(object_pixels)]
    valid = valid[valid > 0]

    avg_depth = np.mean(valid)
    median_depth = np.median(valid)
    print(median_depth)