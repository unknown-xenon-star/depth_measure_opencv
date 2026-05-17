import cv2
import numpy as np

# Load large image

# View window size
win_w, win_h = 1200, 800

# Top-left corner of viewport
offset_x = 0
offset_y = 0

dragging = False
start_x = 0
start_y = 0

def get_pixel(img):
    

    # Resize factor
    scale = 0.25

    # Resize for display
    display_img = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )
    
    display = display_img.copy()

    def mouse_callback(event, x, y, flags, param):
            # if event == cv2.EVENT_MOUSEMOVE:
            if event == cv2.EVENT_LBUTTONDOWN:

                    # Convert back to original coordinates
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)

                    print(f"Display: ({x}, {y})")
                    print(f"Original: ({orig_x}, {orig_y})")

                    display[:] = display_img.copy()
                    # Draw current offset
                    # White outline
                    cv2.putText(
                        display,
                        f"Offset: ({orig_x}, {orig_y})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        5
                    )

                    # Red text
                    cv2.putText(
                        display,
                        f"Offset: ({orig_x}, {orig_y})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                    # Optional marker
                    cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
        
    cv2.namedWindow("Viewer")

    # Attach callback ONCE
    cv2.setMouseCallback("Viewer", mouse_callback)

    while True:

        cv2.imshow("Viewer", display)

        key = cv2.waitKey(20)

        if key & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()