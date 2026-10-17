# Video Face Recognition & Cross-Video Clustering Pipeline

An intelligent, lightweight, and robust Python system that extracts human faces across one or multiple videos, recognizes unique identities using deep neural network embeddings, automatically clusters them into organized folders by person ID and source video, provides one-click jump-to-timestamp links for every face, and visualizes connections between videos and people through conceptual mind maps.

## NB! This web solution has a MIT license
This WEB solution uses follwing third-party AI models:
- YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" (has MIT license)
- SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" (has apache 2.0 license)

DO NOT ALTER, IN ANY WAY, THIS MIT LICENSE (FOR THIS WEB SOLUTION) - IT MAY GIVE LEGAL ISSUES WITH PEOPLE BEHIND MODELS, GIVEN IN URL ABOVE!


---

## 🤖 Important Note on AI-Generated Content / AI Disclosure

> [!IMPORTANT]
> **All images and videos in this project are synthetic and AI-generated with Google Gemini.**
> 
> * **Images:** Portrait images in `data/` (`person_a.jpg`, `person_b.jpg`, `person_c.jpg`).
> * **Videos:** `gemini_generated_video_d6f2b3bb.mp4`, `sample_test_video.mp4`, `sample_videos/*.mp4`, and the multi-video dataset in [`videos/`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/tree/main/videos) ([`gemini_generated_video_2205156f.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_2205156f.mp4), [`gemini_generated_video_662b64d0.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_662b64d0.mp4), [`gemini_generated_video_ab8243d5.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_ab8243d5.mp4)).
> 
> None of the faces or video clips depict real persons; all demonstration material was generated using Gemini.

---

## 🌟 Key Features

1. **Cross-Video Clustering & Analysis**:
   - Scan multiple videos simultaneously (`--videos vid1.mp4 vid2.mp4` or `--video-dir folder/`).
   - Recognizes the same person across different video files and links identities together.
   - Automatically highlights **cross-video persons** who appear in two or more videos.

2. **Clean & Structured Folder Organization**:
   - The output directory is organized by identity, and subsequently by source video:
     `person_000/gemini_generated_video_2205156f/` and `person_000/gemini_generated_video_662b64d0/`.
   - Instantly see in your file explorer which videos each person appears in.

3. **Direct Jump-to-Timestamp Links for Every Face**:
   - For every extracted face image, an accompanying HTML quick-jump file (`face_..._jump.html`) is generated.
   - Double-click the HTML file in Windows Explorer or macOS Finder to open the video directly in your browser, seeked precisely to the exact second and frame where the face was detected!
   - Full CSV export (`timestamps.csv`) with exact timestamp in seconds, formatted time (`00:01.400`), frame number, and direct jump link.

4. **Mind Map Visualization**:
   - **Static High-Resolution Diagram (`mindmap_diagram.png`)**: Displays source videos (color-coded cards) and individuals (circular photo avatars) connected with weighted arcs and time intervals. Cross-video individuals are highlighted with a gold border.
   - **Interactive Web Dashboard (`index.html`)**: Dynamic canvas with physics simulation (drag, zoom, click nodes), integrated video player with timeline, face gallery, and cross-video appearance matrix.

5. **Convenient Launcher Scripts for Windows and Linux/macOS**:
   - `run_clusterer.ps1`: Interactive menu, automatic virtual environment detection, and automatic browser launch for the generated dashboard.
   - `run_clusterer.sh`: Bash equivalent for Linux and macOS.

---

## Examples

<img width="945" height="481" alt="image" src="https://github.com/user-attachments/assets/150a3e51-b101-4cb5-83f9-9a7bd0c1cdd7" />
<img width="941" height="486" alt="image" src="https://github.com/user-attachments/assets/63e96b0f-2f22-47b8-bfbc-1b25ba322f17" />
<img width="935" height="488" alt="image" src="https://github.com/user-attachments/assets/013d39bd-4f60-409d-afd3-9d5433443011" />
<img width="920" height="485" alt="image" src="https://github.com/user-attachments/assets/415a14c3-b99c-469e-a734-665b1c637460" />
<img width="906" height="476" alt="image" src="https://github.com/user-attachments/assets/b2b6bf6e-d867-4b8c-af4c-cdaaa93aedfa" />
<img width="941" height="484" alt="image" src="https://github.com/user-attachments/assets/cbbb66cd-57a4-4da3-b1f8-72d0c4bfb0fe" />

## 📊 Results & Example: Cross-Video Analysis of 3 Videos (`cross_video_output/`)

This example was executed directly against the test dataset located in the repository's [**`videos/`**](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/tree/main/videos) directory.

Run with:
```powershell
# Directly with Python:
python main.py --video-dir videos --output cross_video_output --frame-interval 5

# Or conveniently with the PowerShell helper script:
.\run_clusterer.ps1 -VideoDir "videos" -Output "cross_video_output"

# Or on Linux / macOS:
./run_clusterer.sh --dir "videos" --output "cross_video_output"
```

### 1. Visual Mind Map
The diagram illustrates the connections between the three videos and identified persons. Gold borders indicate **cross-video persons** who appear across multiple videos, while edge labels specify observation counts and time intervals:

<img width="1890" height="961" alt="image" src="https://github.com/user-attachments/assets/e352a4b2-6373-4a07-a73c-4a2cc68902a3" />

---

### 2. Discovered Identities (12 Persons)
The algorithm automatically discovered and separated 12 distinct individual identities across the three videos:

![Discovered identities](cross_video_output/identities_montage.jpg)

---

### 3. Key Analysis Metrics

* **Source videos analyzed from [videos/](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/tree/main/videos):**
  * [`gemini_generated_video_2205156f.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_2205156f.mp4) (11.5 MB — 99 faces detected)
  * [`gemini_generated_video_662b64d0.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_662b64d0.mp4) (2.5 MB — 104 faces detected)
  * [`gemini_generated_video_ab8243d5.mp4`](https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/blob/main/videos/gemini_generated_video_ab8243d5.mp4) (6.5 MB — 33 faces detected)
* **Total detected faces:** 236 faces
* **Discovered unique persons:** 12 identity clusters (`person_000` to `person_011`)
* **Unclustered noise:** Only 2 faces (`unclustered/` - less than 1% noise)
* **⭐ Cross-video persons (found across multiple videos):** **7 persons**
  * `person_000`: 31 faces — found in **all 3 videos** (13 in vid 1, 12 in vid 2, 6 in vid 3)
  * `person_001`: 6 faces — found in **2 videos** (3 in vid 1, 3 in vid 3)
  * `person_002`: 15 faces — found in **all 3 videos** (5 in vid 1, 5 in vid 2, 5 in vid 3)
  * `person_003`: 41 faces — found in **all 3 videos** (17 in vid 1, 17 in vid 2, 7 in vid 3)
  * `person_004`: 35 faces — found in **all 3 videos** (17 in vid 1, 12 in vid 2, 6 in vid 3)
  * `person_005`: 41 faces — found in **all 3 videos** (14 in vid 1, 22 in vid 2, 5 in vid 3)
  * `person_006`: 14 faces — found in **2 videos** (7 in vid 1, 7 in vid 2)
* **Single-video persons:** 5 persons
  * `person_007`: 22 faces (only in `gemini_generated_video_2205156f`)
  * `person_008`: 6 faces (only in `gemini_generated_video_662b64d0`)
  * `person_009`: 7 faces (only in `gemini_generated_video_662b64d0`)
  * `person_010`: 8 faces (only in `gemini_generated_video_662b64d0`)
  * `person_011`: 8 faces (only in `gemini_generated_video_662b64d0`)
* **Silhouette Score:** `0.4903`

---

### 4. Folder Structure in `cross_video_output/`

The structure is organized by person and then source video, complete with dedicated quick-jump HTML files:

```text
cross_video_output/
├── mindmap_diagram.png          # High-resolution visual mind map diagram
├── index.html                   # Interactive web dashboard with graph and video player
├── timestamps.csv               # Complete table of all 236 observations and jump links
├── identities_montage.jpg       # Composite montage image of all 12 identities
├── metadata.json                # Full machine-readable metadata
├── person_000/                  # ⭐ CROSS-VIDEO IDENTITY (in all 3 videos)
│   ├── gemini_generated_video_2205156f/
│   │   ├── face_0001_frame000015_t000.62s.jpg
│   │   └── face_0001_frame000015_t000.62s_jump.html  <- Double-click for direct jump to 00:00.625!
│   ├── gemini_generated_video_662b64d0/
│   │   ├── face_0001_frame000015_t000.62s.jpg
│   │   └── face_0001_frame000015_t000.62s_jump.html
│   └── gemini_generated_video_ab8243d5/
│       ├── face_0001_frame000015_t000.62s.jpg
│       └── face_0001_frame000015_t000.62s_jump.html
├── person_001/                  # ⭐ CROSS-VIDEO IDENTITY (in 2 videos)
│   ├── gemini_generated_video_2205156f/
│   └── gemini_generated_video_ab8243d5/
├── person_002/ ... person_006/  # ⭐ Additional cross-video persons
├── person_007/ ... person_011/  # Single-video persons
└── unclustered/                 # 2 blurry faces (noise < 1%)
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/roblar658/faceImagesExtractAndClusterFromVideo.git
   cd faceImagesExtractAndClusterFromVideo
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv

   # On Windows:
   .\.venv\Scripts\activate
   pip install -r requirements.txt

   # On Linux/macOS:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 🚀 Easy with PowerShell (Windows):
Run the script directly for an interactive menu or pass parameters:
```powershell
# Interactive menu (allows choosing videos or running the demo directly):
.\run_clusterer.ps1

# Process the test directory with the 3 included videos:
.\run_clusterer.ps1 -VideoDir "videos" -Output "cross_video_output"

# Process specific videos:
.\run_clusterer.ps1 -Videos "video1.mp4", "video2.mp4" -Output "my_results"

# Quick synthetic demo run (2 videos sharing a person):
.\run_clusterer.ps1 -Demo
```
*The script automatically detects the virtual environment (`.venv`), runs the analysis, and opens the completed dashboard (`index.html`) automatically in your default browser!*

---

### 🐧 Easy with Unix / macOS:
Make the script executable and run:
```bash
chmod +x run_clusterer.sh

# Interactive menu:
./run_clusterer.sh

# Quick synthetic demo:
./run_clusterer.sh --demo

# Scan directory with the 3 included videos:
./run_clusterer.sh --dir "videos" --output "cross_video_output"
```

---

### 🐍 Directly with Python:

```bash
# Process an entire folder of videos (e.g., the included videos/ directory):
python main.py --video-dir "videos" --output "cross_video_output" --frame-interval 5

# Provide multiple video files directly:
python main.py --videos "videos/gemini_generated_video_2205156f.mp4" "videos/gemini_generated_video_662b64d0.mp4" --output "cross_results"

# Single video:
python main.py --video "gemini_generated_video_d6f2b3bb.mp4" --output "results" --frame-interval 5
```

---

## ⚙️ CLI Arguments

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` / `--videos` | strings | *optional* | One or more paths to video files (also supports wildcards like `'vids/*.mp4'`). |
| `--video-dir` | string | `None` | Directory containing videos (automatically scans for all `.mp4`, `.avi`, `.mkv`, etc.). |
| `--output` | string | `face_clusters_output` | Output folder where subdirectories, images, jump files, mind map, and summary are stored. |
| `--frame-interval` | int | `5` | Samples every N-th frame (e.g., 5 = ~5-6 frames/sec at 25-30 fps). |
| `--method` | choice | `dbscan` | Clustering algorithm: `dbscan` (density-based) or `agglomerative` (hierarchical). |
| `--eps` | float | `0.45` | Cosine distance threshold (lower = stricter, higher = merges more). |
| `--min-samples` | int | `2` | Minimum number of samples to form a valid person cluster in DBSCAN. |
| `--score-thresh` | float | `0.8` | Minimum detection confidence threshold from YuNet (0.0 to 1.0). |
| `--min-face-size` | int | `40` | Minimum pixel size of face bounding box (filters out tiny background faces). |
| `--crop-margin` | float | `0.25` | Extra margin (percentage) around face crops for natural portraits (0.25 = 25%). |
| `--use-aligned` | flag | `False` | Save 112x112 aligned face crops instead of natural portrait crops. |
| `--max-frames` | int | `None` | Maximum number of frames per video (convenient for quick testing). |
| `--models-dir` | string | `models` | Directory for automatic downloading and caching of ONNX models. |

---

## 📂 Project Structure

```text
├── run_clusterer.ps1                   # Convenient PowerShell runner with menu and browser auto-launch
├── run_clusterer.sh                    # Convenient Bash runner for Linux and macOS
├── main.py                             # CLI entry point with cross-video and multi-video support
├── clusterer.py                        # Detection, 128-d SFace embeddings & cross-video clustering
├── visualizer.py                       # Mind map generator (PNG), interactive web dashboard & jump files
├── generate_sample_video.py            # Generator for synthetic cross-video test datasets
├── requirements.txt                    # Required Python packages (OpenCV, scikit-learn, networkx, matplotlib, Pillow)
├── README.md                           # Documentation, benchmark results, and AI disclosure
├── LICENSE                             # MIT License
├── data/                               # AI-generated reference portraits (Gemini)
│   ├── person_a.jpg
│   ├── person_b.jpg
│   └── person_c.jpg
├── sample_videos/                      # Generated synthetic test videos (2 videos sharing a person)
│   ├── video_intervju_del1.mp4
│   └── video_intervju_del2.mp4
├── videos/                             # Multi-video test set (Gemini-generated): https://github.com/roblar658/faceImagesExtractAndClusterFromVideo/tree/main/videos
│   ├── gemini_generated_video_2205156f.mp4
│   ├── gemini_generated_video_662b64d0.mp4
│   └── gemini_generated_video_ab8243d5.mp4
└── cross_video_output/                 # Results from cross-video analysis (3 videos)
    ├── mindmap_diagram.png             # Graphical mind map diagram with nodes and connections
    ├── index.html                      # Interactive dashboard with video player & jump links
    ├── timestamps.csv                  # Tabular overview of all 236 timestamps
    ├── identities_montage.jpg          # Visual montage of all 12 identities
    ├── metadata.json                   # Detailed machine-readable JSON metadata
    ├── person_000/ ... person_006/     # ⭐ Cross-video persons (organized by source video)
    ├── person_007/ ... person_011/     # Single-video persons
    └── unclustered/                    # Unclustered noise
```
