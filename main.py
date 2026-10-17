"""
Command-line interface for video face recognition and clustering.

Usage:
  python main.py --video path/to/video.mp4 --output results/
"""

import os
import sys
import argparse
from clusterer import VideoFaceClusterer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract, recognize, and cluster faces from a video into organized folders by person ID."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g., .mp4, .avi, .mkv, .mov).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="face_clusters_output",
        help="Directory where clustered face folders, metadata, and montage will be saved (default: face_clusters_output).",
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
        help="Maximum total frames to process from the video (useful for quick previews or testing).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to store/find ONNX models (default: models).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 65)
    print("   VIDEO FACE RECOGNITION & CLUSTERING PIPELINE")
    print("=" * 65)

    clusterer = VideoFaceClusterer(
        models_dir=args.models_dir,
        score_threshold=args.score_thresh,
        min_face_size=args.min_face_size,
        crop_margin=args.crop_margin,
    )

    # 1. Detect faces & extract embeddings
    face_records = clusterer.process_video(
        video_path=args.video,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
    )

    if not face_records:
        print("[!] No faces were detected in the video with the current thresholds.")
        print("    Tips: Try lowering --score-thresh (e.g. 0.6) or --min-face-size (e.g. 25).")
        sys.exit(0)

    # 2. Cluster faces by identity
    print(f"\n[*] Clustering {len(face_records)} face embeddings using {args.method.upper()}...")
    clustering_stats = clusterer.cluster_faces(
        face_records=face_records,
        method=args.method,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    print(f"[+] Discovered {clustering_stats['num_clusters']} distinct person identity cluster(s).")
    if clustering_stats.get("noise_count"):
        print(f"    Unclustered noise/outliers: {clustering_stats['noise_count']} face(s).")
    if clustering_stats.get("silhouette_score") is not None:
        print(f"    Cluster Silhouette Score: {clustering_stats['silhouette_score']} (-1 to +1, higher is tighter separation)")

    # 3. Save faces into folders by person ID
    clusterer.save_results(
        face_records=face_records,
        output_dir=args.output,
        stats=clustering_stats,
        use_aligned=args.use_aligned,
    )

    print("\n" + "=" * 65)
    print(f"DONE! All face pictures are categorized and saved in: {os.path.abspath(args.output)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
