from time import time
import cv2
import numpy as np

from tools.disparity_map import disparity_n_depth_map
from tools.hsv import add_HSV_filter
from tools.detection import find_object
from tools.extras import masked_percentile_depth

from tools.config import (
    FOCAL_LENGTH,
    BASELINE,
    MASK_HSV
)

# =========================================================
# CREATE Q MATRIX
# =========================================================

def create_Q_matrix(width, height, focal_length, baseline):

    cx = width / 2
    cy = height / 2

    Q = np.float32([
        [1, 0, 0, -cx],
        [0, -1, 0,  cy],
        [0, 0, 0, -focal_length],
        [0, 0, 1 / baseline, 0]
    ])

    return Q


# =========================================================
# MAIN
# =========================================================

def main():
    cap = cv2.VideoCapture("assest/video.mp4")

    if not cap.isOpened():
        raise RuntimeError("Unable to open video source: assets/video.mp4")

    frame_count = 0
    time_start = time()

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # =====================================================
            # SPLIT LEFT / RIGHT
            # =====================================================

            h, w = frame.shape[:2]
            half = w // 2

            frame_left = frame[:, :half]
            frame_right = frame[:, half:]

            # =====================================================
            # HSV FILTER
            # =====================================================

            mask_left = add_HSV_filter(frame_left, 7, MASK_HSV)
            mask_right = add_HSV_filter(frame_right, 7, MASK_HSV)

            # =====================================================
            # OBJECT DETECTION
            # =====================================================

            circles_left, corr_l = find_object(frame_left, mask_left)
            circles_right, corr_r = find_object(frame_right, mask_right)

            # =====================================================
            # RAW DISPARITY
            # IMPORTANT:
            # False -> raw disparity
            # True  -> visualization
            # =====================================================

            disparity_map, depth_map = disparity_n_depth_map(
                mask_left,
                mask_right,
                False
            )

            # =====================================================
            # FIX DISPARITY FORMAT
            # =====================================================

            if len(disparity_map.shape) == 3:
                disparity_map = cv2.cvtColor(
                    disparity_map,
                    cv2.COLOR_BGR2GRAY
                )

            disparity_map = disparity_map.astype(np.float32)
            disparity_map[disparity_map <= 0] = 0.1

            # =====================================================
            # CREATE Q MATRIX
            # =====================================================

            Q = create_Q_matrix(
                half,
                h,
                FOCAL_LENGTH,
                BASELINE
            )

            # =====================================================
            # REPROJECT TO 3D
            # =====================================================

            points_3D = cv2.reprojectImageTo3D(
                disparity_map,
                Q
            )

            # =====================================================
            # CREATE 3D VIEW WINDOW
            # =====================================================

            vis_3d = np.zeros((h, half, 3), dtype=np.uint8)

            cv2.line(
                vis_3d,
                (half // 2, 0),
                (half // 2, h),
                (100, 100, 100),
                1
            )

            cv2.line(
                vis_3d,
                (0, h // 2),
                (half, h // 2),
                (100, 100, 100),
                1
            )

            # =====================================================
            # TRACKING
            # =====================================================

            if circles_left is None or circles_right is None:
                cv2.putText(
                    frame_right,
                    "TRACKING LOST",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            else:

                # `find_object` coordinates may be floats; convert after clipping
                # so they are valid NumPy indices into the 3D point map.
                coords = np.asarray(corr_r).reshape(-1)

                if coords.size < 2:
                    raise ValueError(f"Invalid object coordinates: {corr_r!r}")

                x = int(np.clip(coords[0], 0, points_3D.shape[1] - 1))
                y = int(np.clip(coords[1], 0, points_3D.shape[0] - 1))

                # =================================================
                # GET 3D POINT
                # =================================================

                X, Y, Z = points_3D[y, x]

                # =================================================
                # OPTIONAL DEPTH
                # =================================================

                depth = masked_percentile_depth(
                    depth_map,
                    mask_right,
                    corr_r,
                    50
                )

                # =================================================
                # DRAW INFO
                # =================================================

                cv2.putText(
                    frame_right,
                    f"X: {X:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame_right,
                    f"Y: {Y:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame_right,
                    f"Z: {Z:.2f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if depth is not None:
                    cv2.putText(
                        frame_right,
                        f"Depth: {depth:.2f} cm",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # =================================================
                # 3D VISUALIZATION
                # =================================================

                scale = 2

                screen_x = int((half // 2) + (X * scale))
                screen_y = int((h // 2) - (Y * scale))

                screen_x = int(np.clip(screen_x, 0, half - 1))
                screen_y = int(np.clip(screen_y, 0, h - 1))

                radius = max(5, int(300 / (abs(Z) + 1)))

                cv2.circle(
                    vis_3d,
                    (screen_x, screen_y),
                    radius,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    vis_3d,
                    "3D OBJECT",
                    (screen_x + 10, screen_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                print("\n========================")
                print("PIXEL:", (x, y))
                print(f"X = {X:.2f}")
                print(f"Y = {Y:.2f}")
                print(f"Z = {Z:.2f}")
                print("========================")

            # =====================================================
            # DISPARITY VISUALIZATION
            # =====================================================

            disp_vis = cv2.normalize(
                disparity_map,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            disp_vis = np.uint8(disp_vis)

            # =====================================================
            # SHOW WINDOWS
            # =====================================================

            cv2.imshow("LEFT CAMERA", frame_left)
            cv2.imshow("RIGHT CAMERA", frame_right)
            cv2.imshow("DISPARITY MAP", disp_vis)
            cv2.imshow("3D VIEW", vis_3d)

            # =====================================================
            # EXIT
            # =====================================================

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        time_diff = max(time() - time_start, 1e-9)

        print(f"""
#| ====================
#| FRAME DATA
#|
#| TOTAL FRAMES: {frame_count}
#| TOTAL TIME: {time_diff:.2f}
#| FPS: {frame_count / time_diff:.2f}
#| ====================
""")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
