"""
Core face detection, recognition, and video clustering pipeline.
Uses OpenCV YuNet for face detection, SFace for 128-d embedding extraction,
and Scikit-Learn for density-based (DBSCAN) and hierarchical (Agglomerative) face clustering.
Supports cross-video clustering, video-segmented folder hierarchies, timestamp jumping,
and mind map visualizations.
"""

import os
import sys
import json
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

import visualizer


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def ensure_model_files(models_dir: str = "models") -> Tuple[str, str]:
    """Ensures YuNet detector and SFace recognizer ONNX models exist locally.
    
    Downloads them automatically from OpenCV Zoo if missing.
    """
    os.makedirs(models_dir, exist_ok=True)
    yunet_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

    if not os.path.exists(yunet_path) or os.path.getsize(yunet_path) < 10000:
        print(f"[*] Downloading YuNet face detection model to {yunet_path}...")
        urllib.request.urlretrieve(YUNET_URL, yunet_path)
        print("    YuNet downloaded successfully.")

    if not os.path.exists(sface_path) or os.path.getsize(sface_path) < 10000:
        print(f"[*] Downloading SFace recognition model to {sface_path}...")
        urllib.request.urlretrieve(SFACE_URL, sface_path)
        print("    SFace downloaded successfully.")

    return yunet_path, sface_path


class FaceRecord:
    """Represents a single face detected in a video frame."""

    def __init__(
        self,
        frame_idx: int,
        timestamp_sec: float,
        bbox: List[int],
        confidence: float,
        raw_detection: np.ndarray,
        aligned_crop: np.ndarray,
        portrait_crop: np.ndarray,
        embedding: np.ndarray,
        video_path: str = "",
        video_name: str = "",
    ):
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = float(confidence)
        self.raw_detection = raw_detection  # 15-dim YuNet output
        self.aligned_crop = aligned_crop    # 112x112 normalized face
        self.portrait_crop = portrait_crop  # natural aspect crop with padding
        self.embedding = embedding          # 128-dim feature vector
        self.video_path = video_path
        self.video_name = video_name
        self.cluster_id: int = -1
        self.saved_path: Optional[str] = None
        self.jump_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "timestamp_hms": visualizer.format_seconds_to_hms(self.timestamp_sec),
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "cluster_id": int(self.cluster_id),
            "saved_path": self.saved_path,
            "jump_file_path": self.jump_file_path,
        }


class VideoFaceClusterer:
    """Extracts, recognizes, and clusters faces from one or multiple videos into organized folders."""

    def __init__(
        self,
        models_dir: str = "models",
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        min_face_size: int = 40,
        crop_margin: float = 0.25,
    ):
        self.models_dir = models_dir
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.crop_margin = crop_margin

        yunet_path, sface_path = ensure_model_files(models_dir)
        self.detector = cv2.FaceDetectorYN.create(
            yunet_path,
            "",
            (320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    def process_video(
        self,
        video_path: str,
        frame_interval: int = 5,
        max_frames: Optional[int] = None,
    ) -> List[FaceRecord]:
        """Reads a single video, detects faces across sampled frames, and extracts embeddings."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.detector.setInputSize((width, height))

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"[*] Processing video: {os.path.basename(video_path)} ({video_name})")
        print(f"    Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames}")
        print(f"    Sampling every {frame_interval} frame(s)")

        frames_to_process = total_frames
        if max_frames and max_frames < total_frames:
            frames_to_process = max_frames

        face_records: List[FaceRecord] = []
        frame_idx = 0

        pbar = tqdm(total=frames_to_process, desc=f"Scanning {os.path.basename(video_path)}", unit="frame")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                _, detections = self.detector.detect(frame)

                if detections is not None:
                    for det in detections:
                        x, y, w, h = [int(v) for v in det[0:4]]
                        conf = float(det[14])

                        # Filter tiny faces or out of bounds
                        if w < self.min_face_size or h < self.min_face_size:
                            continue

                        # Generate aligned face for recognizer (112x112)
                        aligned = self.recognizer.alignCrop(frame, det)
                        if aligned is None or aligned.size == 0:
                            continue

                        # Extract 128-d embedding
                        embedding = self.recognizer.feature(aligned)
                        if embedding is None:
                            continue

                        # Extract natural portrait crop with margin
                        pad_w = int(w * self.crop_margin)
                        pad_h = int(h * self.crop_margin)
                        x1 = max(0, x - pad_w)
                        y1 = max(0, y - pad_h)
                        x2 = min(width, x + w + pad_w)
                        y2 = min(height, y + h + pad_h)
                        portrait = frame[y1:y2, x1:x2].copy()

                        record = FaceRecord(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp,
                            bbox=[x, y, w, h],
                            confidence=conf,
                            raw_detection=det,
                            aligned_crop=aligned,
                            portrait_crop=portrait,
                            embedding=embedding.flatten(),
                            video_path=os.path.abspath(video_path),
                            video_name=video_name,
                        )
                        face_records.append(record)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        print(f"[+] Detection complete for {video_name}. Found {len(face_records)} face(s).")
        return face_records

    def process_videos(
        self,
        video_paths: List[str],
        frame_interval: int = 5,
        max_frames: Optional[int] = None,
    ) -> List[FaceRecord]:
        """Scans multiple videos and aggregates all detected face records for cross-video clustering."""
        all_face_records: List[FaceRecord] = []
        print(f"\n[*] Starting batch processing for {len(video_paths)} video(s)...")

        for idx, vpath in enumerate(video_paths, 1):
            print(f"\n--- [Video {idx}/{len(video_paths)}]: {vpath} ---")
            records = self.process_video(
                video_path=vpath,
                frame_interval=frame_interval,
                max_frames=max_frames,
            )
            all_face_records.extend(records)

        print(f"\n[+] Total faces gathered across all {len(video_paths)} video(s): {len(all_face_records)}")
        return all_face_records

    def cluster_faces(
        self,
        face_records: List[FaceRecord],
        method: str = "dbscan",
        eps: float = 0.45,
        min_samples: int = 2,
    ) -> Dict[str, Any]:
        """Clusters extracted face embeddings across all videos using DBSCAN or Agglomerative Clustering."""
        if not face_records:
            return {"num_clusters": 0, "labels": [], "silhouette": None}

        # Stack and normalize embeddings (L2 norm)
        raw_matrix = np.array([r.embedding for r in face_records], dtype=np.float32)
        norm_matrix = normalize(raw_matrix, norm="l2", axis=1)

        if len(face_records) == 1:
            face_records[0].cluster_id = 0
            return {
                "num_clusters": 1,
                "labels": [0],
                "noise_count": 0,
                "silhouette": None,
                "cross_video_count": 0,
            }

        method = method.lower()
        if method == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            labels = clusterer.fit_predict(norm_matrix)
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(
                metric="cosine",
                linkage="average",
                distance_threshold=eps,
                n_clusters=None,
            )
            labels = clusterer.fit_predict(norm_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}. Choose 'dbscan' or 'agglomerative'.")

        for record, label in zip(face_records, labels):
            record.cluster_id = int(label)

        unique_labels = set(labels)
        noise_count = int(np.sum(labels == -1)) if -1 in unique_labels else 0
        valid_cluster_labels = [l for l in unique_labels if l != -1]
        num_clusters = len(valid_cluster_labels)

        # Calculate silhouette score for quality evaluation if valid
        sil_score = None
        if num_clusters >= 2:
            valid_mask = labels != -1
            if np.sum(valid_mask) > num_clusters:
                try:
                    sil_score = float(silhouette_score(norm_matrix[valid_mask], labels[valid_mask], metric="cosine"))
                except Exception:
                    sil_score = None

        # Cross-video occurrences analysis
        person_videos: Dict[int, set] = {}
        for r in face_records:
            if r.cluster_id != -1:
                person_videos.setdefault(r.cluster_id, set()).add(r.video_name)

        cross_video_clusters = [cid for cid, vset in person_videos.items() if len(vset) > 1]

        stats = {
            "num_clusters": num_clusters,
            "noise_count": noise_count,
            "total_faces": len(face_records),
            "cross_video_count": len(cross_video_clusters),
            "cross_video_clusters": sorted(cross_video_clusters),
            "method": method,
            "eps": eps,
            "min_samples": min_samples,
            "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
        }
        return stats

    def save_results(
        self,
        face_records: List[FaceRecord],
        output_dir: str,
        stats: Dict[str, Any],
        use_aligned: bool = False,
        video_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Saves cropped faces into organized ID and video subfolders, generates jump HTML files,
        exports CSV, generates a static mind map diagram, and creates an interactive dashboard.
        """
        os.makedirs(output_dir, exist_ok=True)

        if video_paths is None:
            # Extract distinct video paths from face records
            vpaths = list({r.video_path for r in face_records if r.video_path})
            video_paths = sorted(vpaths)

        video_names = [os.path.splitext(os.path.basename(vp))[0] for vp in video_paths]

        cluster_counts: Dict[int, int] = {}
        for record in face_records:
            cid = record.cluster_id
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        print(f"\n[*] Saving clustered faces into organized hierarchy in: {os.path.abspath(output_dir)}")

        # 1. Save individual face images and per-face companion jump HTML files
        # Directory structure: output_dir / person_XXX / video_name / face_...jpg & face_..._jump.html
        file_counters: Dict[Tuple[int, str], int] = {}
        rep_face_records: Dict[int, FaceRecord] = {}

        for record in face_records:
            cid = record.cluster_id
            vname = record.video_name or "unknown_video"
            key = (cid, vname)
            file_counters[key] = file_counters.get(key, 0) + 1
            idx_in_person_video = file_counters[key]

            # Determine folder
            folder_person = f"person_{cid:03d}" if cid != -1 else "unclustered"
            target_subfolder = os.path.join(output_dir, folder_person, vname)
            os.makedirs(target_subfolder, exist_ok=True)

            # Face image filename
            img_filename = f"face_{idx_in_person_video:04d}_frame{record.frame_idx:06d}_t{record.timestamp_sec:06.2f}s.jpg"
            img_save_path = os.path.join(target_subfolder, img_filename)

            # Write crop
            img_to_save = record.aligned_crop if use_aligned else record.portrait_crop
            cv2.imwrite(img_save_path, img_to_save)
            record.saved_path = os.path.relpath(img_save_path, output_dir)

            # Generate individual HTML jump file right beside the image
            jump_filename = f"face_{idx_in_person_video:04d}_frame{record.frame_idx:06d}_t{record.timestamp_sec:06.2f}s_jump.html"
            jump_save_path = os.path.join(target_subfolder, jump_filename)

            # Relative path from jump file to source video and to index.html
            video_rel_path = os.path.relpath(record.video_path, target_subfolder).replace("\\", "/")
            index_rel_path = os.path.relpath(os.path.join(output_dir, "index.html"), target_subfolder).replace("\\", "/")

            visualizer.generate_jump_file(
                output_html_path=jump_save_path,
                video_rel_path=video_rel_path,
                image_rel_path=img_filename,
                video_name=vname,
                person_label=folder_person,
                frame_idx=record.frame_idx,
                timestamp_sec=record.timestamp_sec,
                confidence=record.confidence,
                root_index_rel_path=index_rel_path,
            )
            record.jump_file_path = os.path.relpath(jump_save_path, output_dir)

            # Track highest confidence representative face per person
            if cid != -1:
                if cid not in rep_face_records or record.confidence > rep_face_records[cid].confidence:
                    rep_face_records[cid] = record

        # 2. Extract representative crops and relative paths
        representative_faces: Dict[int, np.ndarray] = {
            cid: rec.portrait_crop for cid, rec in rep_face_records.items()
        }
        rep_face_rel_paths: Dict[int, str] = {
            cid: rec.saved_path for cid, rec in rep_face_records.items() if rec.saved_path
        }

        # 3. Export CSV summary of all appearances
        csv_path = os.path.join(output_dir, "timestamps.csv")
        visualizer.export_timestamps_csv(face_records, csv_path)
        print(f"[+] Timestamp references and jump index exported to: {csv_path}")

        # 4. Generate visual montage of discovered identities
        montage_path = self._generate_montage(representative_faces, output_dir)

        # 5. Generate high-resolution static Mind Map diagram (PNG)
        mindmap_img_path = visualizer.generate_mindmap_image(
            face_records=face_records,
            output_dir=output_dir,
            rep_faces=representative_faces,
            video_names=video_names,
            output_filename="mindmap_diagram.png",
        )

        # 6. Generate interactive dashboard with dynamic mind map and built-in video player (HTML)
        dashboard_path = visualizer.generate_interactive_dashboard(
            face_records=face_records,
            output_dir=output_dir,
            stats=stats,
            rep_face_rel_paths=rep_face_rel_paths,
            video_paths=video_paths,
        )

        # 7. Write comprehensive metadata JSON
        # Group breakdown by person and by video
        cluster_breakdown = {}
        for cid in sorted(cluster_counts.keys()):
            person_lbl = f"person_{cid:03d}" if cid != -1 else "unclustered"
            person_faces = [r for r in face_records if r.cluster_id == cid]
            vids_for_person = {}
            for r in person_faces:
                vids_for_person[r.video_name] = vids_for_person.get(r.video_name, 0) + 1

            cluster_breakdown[person_lbl] = {
                "cluster_id": cid,
                "total_faces": len(person_faces),
                "is_cross_video": len(vids_for_person) > 1,
                "video_occurrences": vids_for_person,
                "rep_image": rep_face_rel_paths.get(cid),
            }

        metadata = {
            "clustering_stats": stats,
            "videos": [
                {
                    "name": vname,
                    "path": vp,
                    "total_faces": sum(1 for r in face_records if r.video_name == vname),
                }
                for vname, vp in zip(video_names, video_paths)
            ],
            "cluster_breakdown": cluster_breakdown,
            "faces": [r.to_dict() for r in face_records],
            "montage_path": os.path.relpath(montage_path, output_dir) if montage_path else None,
            "mindmap_image_path": os.path.relpath(mindmap_img_path, output_dir),
            "dashboard_html_path": os.path.relpath(dashboard_path, output_dir),
            "timestamps_csv_path": os.path.relpath(csv_path, output_dir),
        }

        meta_file = os.path.join(output_dir, "metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[+] Processing finished successfully!")
        print(f"    - Discovered Persons: {stats.get('num_clusters', 0)}")
        print(f"    - Cross-Video Persons: {stats.get('cross_video_count', 0)}")
        print(f"    - Mind Map Diagram: {mindmap_img_path}")
        print(f"    - Interactive Dashboard: {dashboard_path}")
        print(f"    - Metadata JSON: {meta_file}")

        return metadata

    def _generate_montage(
        self,
        rep_faces: Dict[int, np.ndarray],
        output_dir: str,
        thumb_size: int = 160,
    ) -> Optional[str]:
        """Creates a visual grid summary showing one representative portrait per person."""
        if not rep_faces:
            return None

        n_persons = len(rep_faces)
        cols = min(4, n_persons)
        rows = (n_persons + cols - 1) // cols

        card_w = thumb_size + 20
        card_h = thumb_size + 50
        grid_w = cols * card_w + 20
        grid_h = rows * card_h + 80

        canvas = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)

        # Header Title
        cv2.putText(
            canvas,
            f"Discovered Identities ({n_persons} Persons)",
            (20, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        for i, (cid, img) in enumerate(rep_faces.items()):
            r = i // cols
            c = i % cols
            x = 20 + c * card_w
            y = 70 + r * card_h

            # Draw white card box with subtle border
            cv2.rectangle(canvas, (x, y), (x + thumb_size, y + card_h - 15), (255, 255, 255), -1)
            cv2.rectangle(canvas, (x, y), (x + thumb_size, y + card_h - 15), (210, 210, 210), 1)

            # Resize face to fit thumb_size x thumb_size
            resized = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
            canvas[y : y + thumb_size, x : x + thumb_size] = resized

            # Label text
            label = f"Person #{cid:03d}"
            cv2.putText(
                canvas,
                label,
                (x + 10, y + thumb_size + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        montage_path = os.path.join(output_dir, "identities_montage.jpg")
        cv2.imwrite(montage_path, canvas)
        return montage_path
