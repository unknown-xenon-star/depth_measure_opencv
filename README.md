# depth_measure_opencv
Depth Meadure Using OPENCV, numpy, mediapipe 
## Project Related to: DRDO


### Basic Input


## Steps 
METHOD WE ARE USING
 - calibrate the cameras
 - input 2 stream of data
 - stereo.compute
    1. Take a small block/window around that pixel
    2. Search horizontally in the right image
    3. Find the most similar block
    4. Compute horizontal shift
 - calculate the disparity map
 - calculate the depth
