"""
Generates realistic synthetic test videos containing distinct people moving across scenes.
Supports generating pairs of videos with overlapping person identities to verify
cross-video clustering, timestamp jumping, and mind map visualizations.
"""

import os
import cv2
import numpy as np


def create_test_video(
    output_video_path="sample_test_video.mp4",
    img_first_path="data/person_a.jpg",
    img_second_path="data/person_b.jpg",
    video_title="Test Video Stream",
    duration_sec=4,
    fps=25,
):
    """Creates a synthetic video containing two people with an overlapping segment."""
    if not os.path.exists(img_first_path) or not os.path.exists(img_second_path):
        raise FileNotFoundError(f"Missing portrait file(s): {img_first_path} or {img_second_path}")

    img_a = cv2.imread(img_first_path)
    img_b = cv2.imread(img_second_path)

    width, height = 800, 600
    total_frames = int(duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[*] Generating test video: {output_video_path}")
    print(f"    Dimensions: {width}x{height} | FPS: {fps} | Frames: {total_frames}")

    p_size = 260
    a_resized = cv2.resize(img_a, (p_size, p_size))
    b_resized = cv2.resize(img_b, (p_size, p_size))

    for frame_i in range(total_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (230, 235, 240)

        # First person: frame 0 to 55
        if frame_i < 55:
            offset_x = int(60 + 30 * np.sin(frame_i / 8.0))
            offset_y = int(140 + 15 * np.cos(frame_i / 8.0))
            canvas[offset_y : offset_y + p_size, offset_x : offset_x + p_size] = a_resized

        # Second person: frame 35 to 100 (frames 35-55 both persons appear)
        if frame_i >= 35:
            offset_x = int(450 + 25 * np.cos(frame_i / 7.0))
            offset_y = int(150 + 20 * np.sin(frame_i / 7.0))
            canvas[offset_y : offset_y + p_size, offset_x : offset_x + p_size] = b_resized

        t_sec = frame_i / fps
        cv2.putText(
            canvas,
            f"{video_title} - Time: {t_sec:.2f}s (Frame {frame_i:03d})",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )

        out.write(canvas)

    out.release()
    print(f"[+] Video successfully saved: {os.path.abspath(output_video_path)}")


def generate_cross_video_dataset():
    """Generates a multi-video test set where Person B appears in BOTH videos (Cross-Video link)."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("sample_videos", exist_ok=True)

    # Video 1: Person A and Person B
    vid1_path = os.path.join("sample_videos", "video_intervju_del1.mp4")
    create_test_video(
        output_video_path=vid1_path,
        img_first_path="data/person_a.jpg",
        img_second_path="data/person_b.jpg",
        video_title="Del 1: Intervju med Person A & B",
        duration_sec=4,
        fps=25,
    )

    # Video 2: Person B (returning!) and Person C
    img_c = "data/person_c.jpg" if os.path.exists("data/person_c.jpg") else "data/person_a.jpg"
    vid2_path = os.path.join("sample_videos", "video_intervju_del2.mp4")
    create_test_video(
        output_video_path=vid2_path,
        img_first_path="data/person_b.jpg",
        img_second_path=img_c,
        video_title="Del 2: Oppfølging med Person B & C",
        duration_sec=4,
        fps=25,
    )

    print("\n[+] Multi-video dataset ready in 'sample_videos/':")
    print(f"    - {vid1_path} (Person A & Person B)")
    print(f"    - {vid2_path} (Person B & Person C)")
    print("    * Notice: Person B is present in BOTH videos (the ground-truth cross-video link)!")


if __name__ == "__main__":
    generate_cross_video_dataset()
