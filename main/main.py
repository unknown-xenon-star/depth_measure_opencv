import cv2 

from config import (
    MODE,

    FOCAL_LENGTH, 
    BASELINE,
    DISPARITY_OFFSET,
    
    LEFT_IMAGE,
    RIGHT_IMAGE 
)

from tools.disparity_map import disparity_n_depth_map
from tools.seperate_layers import save_sepreated_layers, save_images
from tools.detection import segmentation
from tools.get_pixel import get_pixel

def main() -> None:
    
    left = cv2.imread(LEFT_IMAGE)
    right = cv2.imread(RIGHT_IMAGE)

    if left is None or right is None:
        raise FileNotFoundError("Could not load images.")
    
    disparity_map, depth_map = disparity_n_depth_map(left, right, Colored=True)
    # save_images([disparity_map, depth_map], prefix="initial")
    # while True:
    #     # cv2.imshow("disparity_map", cv2.resize(disparity_map, (0, 0), fx=0.35, fy=0.35))
    #     # cv2.imshow("depth_map", cv2.resize(depth_map, (0, 0), fx=0.35, fy=0.35))
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # print("HELLO")
    # save_sepreated_layers(disparity_map, layered=False)
    # cv2.destroyAllWindows()
    # get_pixel(depth_map)
    segmentation(depth_map)

if __name__ == "__main__":
    main()
