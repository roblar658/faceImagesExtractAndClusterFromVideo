"""
Generates a realistic test video containing two distinct people moving and co-occurring in scenes.
Uses test portraits to simulate a live video stream for verification.
"""

import os
import cv2
import numpy as np


def create_test_video(output_video_path="sample_test_video.mp4", duration_sec=4, fps=25):
    img_a_path = os.path.join("data", "person_a.jpg")
    img_b_path = os.path.join("data", "person_b.jpg")

    if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
        raise FileNotFoundError("Missing person_a.jpg or person_b.jpg in data/")

    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    # Frame canvas size
    width, height = 800, 600
    total_frames = int(duration_sec * fps)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[*] Generating synthetic test video: {output_video_path}")
    print(f"    Dimensions: {width}x{height} | FPS: {fps} | Frames: {total_frames}")

    # Desired face portrait size on canvas
    p_size = 260
    a_resized = cv2.resize(img_a, (p_size, p_size))
    b_resized = cv2.resize(img_b, (p_size, p_size))

    for frame_i in range(total_frames):
        # Create subtle background gradient
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (230, 235, 240)

        # Person A appears from frame 0 to frame 55 (drifting slightly left to right)
        if frame_i < 55:
            # Person A x position
            offset_x = int(60 + 30 * np.sin(frame_i / 8.0))
            offset_y = int(140 + 15 * np.cos(frame_i / 8.0))
            canvas[offset_y : offset_y + p_size, offset_x : offset_x + p_size] = a_resized

        # Person B appears from frame 35 to frame 100 (frames 35-55 have BOTH people together)
        if frame_i >= 35:
            offset_x = int(450 + 25 * np.cos(frame_i / 7.0))
            offset_y = int(150 + 20 * np.sin(frame_i / 7.0))
            canvas[offset_y : offset_y + p_size, offset_x : offset_x + p_size] = b_resized

        # Add timestamp & scene text
        t_sec = frame_i / fps
        cv2.putText(
            canvas,
            f"Test Video Stream - Time: {t_sec:.2f}s (Frame {frame_i:03d})",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

        out.write(canvas)

    out.release()
    print(f"[+] Sample test video generated successfully: {os.path.abspath(output_video_path)}")


if __name__ == "__main__":
    create_test_video()
