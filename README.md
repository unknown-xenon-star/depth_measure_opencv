# depth_measure_opencv

Depth measurement experiments using OpenCV, NumPy, and stereo image pairs.

## Project Related to

DRDO

## Main Files

- `depth_map.py`: computes disparity and depth from stereo input.
- `depth_click.py`: click on the left image to inspect depth at a pixel.
- `input.py`: selects static image mode or live camera mode.
- `main/main.py`: alternate pipeline for disparity, segmentation, and layer separation.

## Included Assets

- Sample stereo images: `im0.png`, `im1.png`
- Additional images and experiments under `main/tools/data/`, `main/down/`, and `main/assest/`
- Experimental scripts: `output.py`, `The Semi-Humane I .py`, `diff semi.py`, `main/victus.py`, `main/V_1.py`

## Steps

Method used:

- calibrate the cameras
- input 2 streams of data
- run `stereo.compute`
  1. take a small block/window around a pixel
  2. search horizontally in the right image
  3. find the most similar block
  4. compute horizontal shift
- calculate the disparity map
- calculate the depth

