import cv2
import time
import os
from picamera2 import Picamera2

# --- Configuration ---
INTERVAL = 3.0  # Time delay between captures in seconds
OUTPUT_DIR = "captured_images"  # Base folder where images will be saved
LEFT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "left_cam")
RIGHT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "right_cam")

# Create output directories if they don't exist
os.makedirs(LEFT_OUTPUT_DIR, exist_ok=True)
os.makedirs(RIGHT_OUTPUT_DIR, exist_ok=True)

# Initialize Picamera2 for both cameras
print("Initializing left and right Pi cameras...")
left_cam = Picamera2(camera_num=0)
right_cam = Picamera2(camera_num=1)

# Configure camera settings (Optional: Adjust resolution if needed)
# The default configuration usually matches your sensor's profile.
left_config = left_cam.create_preview_configuration()
right_config = right_cam.create_preview_configuration()
left_cam.configure(left_config)
right_cam.configure(right_config)

# Start both camera streams
left_cam.start()
right_cam.start()

print(f"Starting dual image capture every {INTERVAL} seconds...")
print("Press 'q' in either camera window to quit.")

# Initialize the tracking timer
last_capture_time = time.time()
img_counter = 0

try:
    while True:
        # Capture frames from both cameras
        left_frame = left_cam.capture_array()
        right_frame = right_cam.capture_array()

        # Picamera2 captures in RGB by default; OpenCV needs BGR for proper display/saving
        left_frame = cv2.flip(left_frame, 0)
        right_frame = cv2.flip(right_frame, 0)
        left_frame_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        right_frame_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

        # Display both live video feeds
        cv2.imshow("Left Camera Feed (Press Q to Quit)", left_frame_bgr)
        cv2.imshow("Right Camera Feed (Press Q to Quit)", right_frame_bgr)

        # Check if the capture interval has passed
        current_time = time.time()
        if current_time - last_capture_time >= INTERVAL:
            left_img_name = os.path.join(LEFT_OUTPUT_DIR, f"img_{img_counter:03d}.jpg")
            right_img_name = os.path.join(RIGHT_OUTPUT_DIR, f"img_{img_counter:03d}.jpg")

            # Save both images with the same index to keep them paired
            cv2.imwrite(left_img_name, left_frame_bgr)
            cv2.imwrite(right_img_name, right_frame_bgr)
            print(f"Saved left: {left_img_name}")
            print(f"Saved right: {right_img_name}")

            img_counter += 1
            last_capture_time = current_time

        # Listen for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping capture script...")
            break

finally:
    # Clean up and release camera resources cleanly
    left_cam.stop()
    right_cam.stop()
    cv2.destroyAllWindows()
    print("Cameras stopped. Done!")

