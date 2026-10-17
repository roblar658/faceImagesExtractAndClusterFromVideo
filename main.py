"""
Command-line interface for video face recognition, cross-video clustering,
timestamp jumping, and mind map visualizations.

Usage:
  # Single video:
  python main.py --video path/to/video.mp4 --output results/

  # Multiple videos (Cross-video clustering):
  python main.py --videos video1.mp4 video2.mp4 video3.mp4 --output cross_video_results/

  # Whole folder of videos:
  python main.py --video-dir path/to/videos/ --output results/
"""

import os
import sys
import glob
import argparse
from typing import List

# Ensure terminal stdout/stderr uses UTF-8 where available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from clusterer import VideoFaceClusterer


VALID_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract, recognize, and cluster faces across one or multiple videos into organized folders with timestamp jump links and mind map visualization."
    )
    parser.add_argument(
        "--video",
        "--videos",
        dest="videos",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to one or more input video files (e.g., --video v1.mp4 v2.mp4 or glob pattern 'vids/*.mp4').",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing video files to process all videos inside it.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="face_clusters_output",
        help="Directory where clustered face folders, metadata, jump files, and mind map will be saved (default: face_clusters_output).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="Sample 1 frame every N frames to avoid redundant near-identical frames (default: 5).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dbscan", "agglomerative"],
        default="dbscan",
        help="Clustering algorithm: 'dbscan' (density-based with noise handling) or 'agglomerative' (hierarchical). Default: dbscan.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.45,
        help="Cosine distance clustering threshold (lower = stricter/fewer merges, higher = looser/more merges). Default: 0.45.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum faces required to establish a valid identity cluster in DBSCAN. Singletons go to 'unclustered' if > 1 (default: 2).",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.8,
        help="Face detector confidence score threshold between 0.0 and 1.0 (default: 0.8).",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum face width/height in pixels to reject tiny background faces (default: 40).",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.25,
        help="Margin percentage added around the face bounding box for natural portrait crops (default: 0.25).",
    )
    parser.add_argument(
        "--use-aligned",
        action="store_true",
        help="Save 112x112 normalized aligned face crops instead of natural portrait crops.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum total frames to process per video (useful for quick previews or testing).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store/find ONNX models (default: models).",
    )
    return parser.parse_args()


def collect_video_files(video_args: List[str], video_dir: str = None) -> List[str]:
    """Collects and validates video files from arguments and directories."""
    discovered: List[str] = []

    if video_args:
        for item in video_args:
            # Handle possible wildcard glob patterns
            matches = glob.glob(item)
            if matches:
                discovered.extend(matches)
            elif os.path.exists(item):
                discovered.append(item)
            else:
                print(f"[!] Warning: Video path '{item}' not found, skipping.", file=sys.stderr)

    if video_dir:
        if not os.path.exists(video_dir):
            print(f"[!] Warning: Video directory '{video_dir}' does not exist.", file=sys.stderr)
        else:
            for root, _, files in os.walk(video_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in VALID_VIDEO_EXTS:
                        discovered.append(os.path.join(root, f))

    # Deduplicate while preserving order
    unique_videos: List[str] = []
    seen = set()
    for v in discovered:
        norm = os.path.normpath(os.path.abspath(v))
        if norm not in seen and os.path.isfile(norm):
            seen.add(norm)
            unique_videos.append(v)

    return unique_videos


def main():
    args = parse_args()

    video_files = collect_video_files(args.videos, args.video_dir)
    if not video_files:
        print("Error: No valid video files specified. Use --video / --videos or --video-dir.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("   VIDEO FACE RECOGNITION, CLUSTERING & MIND MAP PIPELINE")
    print("=" * 70)
    print(f"[*] Total Videos to process: {len(video_files)}")
    for i, v in enumerate(video_files, 1):
        print(f"    {i}. {v}")
    print(f"[*] Target Output Directory: {os.path.abspath(args.output)}")
    print(f"[*] Sampling frame interval: Every {args.frame_interval} frame(s)")
    print(f"[*] Clustering Algorithm:   {args.method.upper()} (eps={args.eps})")
    print("=" * 70)

    clusterer = VideoFaceClusterer(
        models_dir=args.models_dir,
        score_threshold=args.score_thresh,
        min_face_size=args.min_face_size,
        crop_margin=args.crop_margin,
    )

    # 1. Detect faces & extract embeddings across all videos
    face_records = clusterer.process_videos(
        video_paths=video_files,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )

    if not face_records:
        print("\n[!] No faces were detected across the provided videos with the current thresholds.")
        print("    Tips: Try lowering --score-thresh (e.g. 0.6) or --min-face-size (e.g. 25).")
        sys.exit(0)

    # 2. Cluster faces across all videos
    print(f"\n[*] Clustering {len(face_records)} face embedding(s) using {args.method.upper()}...")
    clustering_stats = clusterer.cluster_faces(
        face_records=face_records,
        method=args.method,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    num_clusters = clustering_stats.get("num_clusters", 0)
    cross_video_count = clustering_stats.get("cross_video_count", 0)
    print(f"[+] Discovered {num_clusters} distinct person identity cluster(s).")
    if cross_video_count > 0:
        print(f"[+] [CROSS-VIDEO] Found {cross_video_count} identity/identities appearing across multiple videos!")
    if clustering_stats.get("noise_count"):
        print(f"    Unclustered noise/outliers: {clustering_stats['noise_count']} face(s).")
    if clustering_stats.get("silhouette_score") is not None:
        print(f"    Cluster Silhouette Score: {clustering_stats['silhouette_score']} (-1 to +1, higher is tighter separation)")

    # 3. Save faces, jump files, CSV, static mind map, and interactive dashboard
    metadata = clusterer.save_results(
        face_records=face_records,
        output_dir=args.output,
        stats=clustering_stats,
        use_aligned=args.use_aligned,
        video_paths=video_files,
    )

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print(f" - Folders & Images:      {os.path.abspath(args.output)}")
    print(f" - Mind Map Diagram (PNG):{os.path.abspath(os.path.join(args.output, 'mindmap_diagram.png'))}")
    print(f" - Interactive Dashboard: {os.path.abspath(os.path.join(args.output, 'index.html'))}")
    print(f" - Timestamps (CSV):      {os.path.abspath(os.path.join(args.output, 'timestamps.csv'))}")
    print(f" - Identity Montage:      {os.path.abspath(os.path.join(args.output, 'identities_montage.jpg'))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
