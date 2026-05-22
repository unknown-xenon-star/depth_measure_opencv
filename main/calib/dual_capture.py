#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture video from two Picamera2 cameras and save side-by-side output."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dual_camera_output.mp4",
        help="Output video filename.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of each camera stream.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of each camera stream.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for capture and output.",
    )
    parser.add_argument(
        "--camera-left",
        type=int,
        default=0,
        help="Picamera2 index for the left camera.",
    )
    parser.add_argument(
        "--camera-right",
        type=int,
        default=1,
        help="Picamera2 index for the right camera.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds. Use 0 to record until Ctrl+C.",
    )
    return parser.parse_args()


def create_camera(camera_num, width, height):
    camera = Picamera2(camera_num=camera_num)
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    return camera


def main():
    args = parse_args()
    output_path = Path(args.output)
    frame_size = (args.width * 2, args.height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, frame_size)

    if not writer.isOpened():
        print(f"Failed to open video writer for: {output_path}", file=sys.stderr)
        return 1

    left_camera = None
    right_camera = None

    try:
        left_camera = create_camera(args.camera_left, args.width, args.height)
        right_camera = create_camera(args.camera_right, args.width, args.height)

        # Let both sensors settle before the first capture.
        time.sleep(1.0)

        start_time = time.time()
        frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0

        print(
            "Recording started. Press Ctrl+C to stop."
            if args.duration == 0
            else f"Recording for {args.duration:.1f} seconds..."
        )

        while True:
            loop_started = time.time()

            left_frame = left_camera.capture_array("main")
            right_frame = right_camera.capture_array("main")

            if left_frame.shape[:2] != (args.height, args.width):
                left_frame = cv2.resize(left_frame, (args.width, args.height))
            if right_frame.shape[:2] != (args.height, args.width):
                right_frame = cv2.resize(right_frame, (args.width, args.height))

            combined_frame = np.hstack((left_frame, right_frame))
            writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

            if frame_interval > 0:
                elapsed = time.time() - loop_started
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        if left_camera is not None:
            left_camera.stop()
        if right_camera is not None:
            right_camera.stop()
        writer.release()

    print(f"Saved video to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

