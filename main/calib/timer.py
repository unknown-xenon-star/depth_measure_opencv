import cv2
import time
import os
from picamera2 import Picamera2

# --- Configuration ---
INTERVAL = 3.0  # Time delay between captures in seconds
OUTPUT_DIR = "captured_images"  # Folder where images will be saved

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize Picamera2
print("Initializing Pi Camera...")
picam = Picamera2()

# Configure camera settings (Optional: Adjust resolution if needed)
# The default configuration usually matches your sensor's profile.
config = picam.create_preview_configuration()
picam.configure(config)

# Start the camera stream
picam.start()

print(f"Starting image capture every {INTERVAL} seconds...")
print("Press 'q' in the camera window to quit.")

# Initialize the tracking timer
last_capture_time = time.time()
img_counter = 0

try:
    while True:
        # Capture a frame as a NumPy array (directly compatible with OpenCV)
        frame = picam.capture_array()
        
        # Picamera2 captures in RGB by default; OpenCV needs BGR for proper display/saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the live video feed
        cv2.imshow('RPi5 Camera Feed (Press Q to Quit)', frame_bgr)

        # Check if 3 seconds have passed
        current_time = time.time()
        if current_time - last_capture_time >= INTERVAL:
            # Generate a unique filename
            img_name = os.path.join(OUTPUT_DIR, f"img_{img_counter:03d}.jpg")
            
            # Save the image
            cv2.imwrite(img_name, frame_bgr)
            print(f"Saved: {img_name}")
            
            # Update counter and reset the timer
            img_counter += 1
            last_capture_time = current_time

        # Listen for the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping capture script...")
            break

finally:
    # Clean up and release camera resources cleanly
    picam.stop()
    cv2.destroyAllWindows()
    print("Camera stopped. Done!")