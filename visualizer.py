"""
Visualization module for video face clustering.
Provides:
1. High-resolution static mind map diagram generation (matplotlib/networkx/PIL).
2. Per-face companion HTML jump files to easily seek directly to the exact video timestamp.
3. Modern, interactive single-page dashboard with an interactive mind map and integrated video player.
4. Tabular CSV timestamps export for cross-video analysis.
"""

import os
import csv
import json
import math
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx


def format_seconds_to_hms(seconds: float) -> str:
    """Formats seconds as HH:MM:SS.mmm or MM:SS.mmm."""
    mins = int(seconds // 60)
    hrs = int(mins // 60)
    mins = mins % 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
    return f"{mins:02d}:{secs:06.3f}"


def make_circular_thumbnail(img_bgr: np.ndarray, size: int = 120, ring_color: tuple = (255, 255, 255), ring_width: int = 4) -> Image.Image:
    """Converts a BGR OpenCV crop to a high-quality circular PIL RGBA image with border."""
    rgb = img_bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    pil_img = ImageOps.fit(pil_img, (size, size), centering=(0.5, 0.5))

    mask = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0, size, size), fill=255)

    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(pil_img, (0, 0), mask)

    if ring_width > 0:
        draw_ring = ImageDraw.Draw(result)
        draw_ring.ellipse(
            (ring_width // 2, ring_width // 2, size - ring_width // 2, size - ring_width // 2),
            outline=ring_color,
            width=ring_width,
        )

    return result


def generate_jump_file(
    output_html_path: str,
    video_rel_path: str,
    image_rel_path: str,
    video_name: str,
    person_label: str,
    frame_idx: int,
    timestamp_sec: float,
    confidence: float,
    root_index_rel_path: str,
) -> None:
    """Generates an individual HTML jump file for a face image.
    
    When double-clicked, it loads the video and jumps directly to timestamp_sec.
    """
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    hms = format_seconds_to_hms(timestamp_sec)

    # HTML template with modern styling and HTML5 video jump controls
    html_content = f"""<!DOCTYPE html>
<html lang="no">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tidshopp: {person_label} | {video_name} @ {hms}</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #151d2f;
      --card-border: #23304a;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.25);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
    }}
    .container {{
      max-width: 900px;
      width: 100%;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 16px;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-group h1 {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #fff;
    }}
    .title-group p {{
      font-size: 0.9rem;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .badge {{
      background: #0284c7;
      color: #fff;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      letter-spacing: 0.5px;
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: 200px 1fr;
      gap: 24px;
    }}
    @media (max-width: 700px) {{
      .content-grid {{ grid-template-columns: 1fr; }}
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      background: rgba(11, 15, 25, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
    }}
    .face-thumb {{
      width: 140px;
      height: 140px;
      border-radius: 12px;
      object-fit: cover;
      border: 3px solid var(--accent);
      box-shadow: 0 0 15px var(--accent-glow);
      margin-bottom: 12px;
    }}
    .meta-item {{
      margin: 6px 0;
      font-size: 0.85rem;
      color: var(--text-muted);
      width: 100%;
      text-align: left;
    }}
    .meta-item strong {{
      color: #e2e8f0;
    }}
    .player-section {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
      max-height: 480px;
    }}
    .controls-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      background: #1e293b;
      color: #fff;
      border: 1px solid #334155;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
    }}
    .btn:hover {{
      background: #334155;
      border-color: #475569;
    }}
    .btn-primary {{
      background: #0284c7;
      border-color: #0369a1;
      font-weight: 600;
    }}
    .btn-primary:hover {{
      background: #0369a1;
    }}
    .back-btn {{
      margin-left: auto;
      background: #3b82f6;
    }}
    .notice {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title-group">
        <h1>Tidspunkt: {hms} ({timestamp_sec:.2f}s)</h1>
        <p>Kildevideo: <strong>{video_name}</strong> | Identitet: <strong>{person_label}</strong></p>
      </div>
      <div class="badge">Ramme #{frame_idx}</div>
    </div>

    <div class="content-grid">
      <div class="sidebar">
        <img class="face-thumb" src="{image_rel_path}" alt="Ansiktsutsnitt">
        <div class="meta-item"><strong>Person:</strong> {person_label}</div>
        <div class="meta-item"><strong>Video:</strong> {video_name}</div>
        <div class="meta-item"><strong>Tidspunkt:</strong> {hms}</div>
        <div class="meta-item"><strong>Sekunder:</strong> {timestamp_sec:.3f}s</div>
        <div class="meta-item"><strong>Ramme (Frame):</strong> #{frame_idx}</div>
        <div class="meta-item"><strong>Konfidens:</strong> {confidence * 100:.1f}%</div>
      </div>

      <div class="player-section">
        <video id="vidPlayer" controls preload="metadata">
          <source src="{video_rel_path}#t={timestamp_sec}" type="video/mp4">
          Nettleseren støtter ikke videoavspilling.
        </video>

        <div class="controls-bar">
          <button class="btn btn-primary" onclick="jumpToExact()">Hopp til {hms} & spill</button>
          <button class="btn" onclick="stepTime(-1)">-1 sek</button>
          <button class="btn" onclick="stepTime(-0.1)">-0.1s</button>
          <button class="btn" onclick="stepTime(0.1)">+0.1s</button>
          <button class="btn" onclick="stepTime(1)">+1 sek</button>
          <a class="btn back-btn" href="{root_index_rel_path}">Hovedoversikt / Tankediagram</a>
        </div>
        <div class="notice">
          Videoen åpnes automatisk ved {hms}. Hvis nettleseren blokkerer autoplay, trykk på <strong>Hopp til {hms} & spill</strong>.
        </div>
      </div>
    </div>
  </div>

  <script>
    const targetTimestamp = {timestamp_sec};
    const player = document.getElementById('vidPlayer');

    function jumpToExact() {{
      player.currentTime = targetTimestamp;
      player.play().catch(e => console.log('Autoplay venter på klikk'));
    }}

    function stepTime(delta) {{
      player.currentTime = Math.max(0, player.currentTime + delta);
    }}

    // Prøv å hoppe til eksakt tidspunkt ved lasting
    player.addEventListener('loadedmetadata', function() {{
      player.currentTime = targetTimestamp;
    }});

    window.addEventListener('DOMContentLoaded', function() {{
      setTimeout(jumpToExact, 150);
    }});
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def export_timestamps_csv(face_records: List[Any], csv_path: str) -> None:
    """Exports a structured CSV summary of all face occurrences with timestamps and jump paths."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "Person",
        "Video",
        "Tidspunkt_Sekunder",
        "Tidspunkt_HMS",
        "Frame_Nummer",
        "Konfidens",
        "Bilde_Filsti",
        "Tidshopp_HTML",
        "Kildevideo_Filsti",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in face_records:
            cid = r.cluster_id
            person_lbl = f"person_{cid:03d}" if cid != -1 else "unclustered"
            hms = format_seconds_to_hms(r.timestamp_sec)
            writer.writerow([
                person_lbl,
                r.video_name,
                f"{r.timestamp_sec:.3f}",
                hms,
                r.frame_idx,
                f"{r.confidence:.4f}",
                r.saved_path or "",
                getattr(r, "jump_file_path", "") or "",
                r.video_path or "",
            ])


def generate_mindmap_image(
    face_records: List[Any],
    output_dir: str,
    rep_faces: Dict[int, np.ndarray],
    video_names: List[str],
    output_filename: str = "mindmap_diagram.png",
) -> str:
    """Generates a high-resolution, publication-ready mind map / network diagram (PNG).
    
    Shows connections between videos and persons, with face thumbnail avatars and
    prominent highlights for cross-video identities.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 1. Aggregate connection statistics: Person <-> Video
    # connections[person_id][video_name] = {"count": N, "min_t": X, "max_t": Y}
    person_video_stats: Dict[int, Dict[str, Dict[str, Any]]] = {}
    video_face_counts: Dict[str, int] = {v: 0 for v in video_names}

    for r in face_records:
        cid = r.cluster_id
        if cid == -1:
            continue  # don't clutter mindmap with unclustered noise
        vname = r.video_name
        video_face_counts[vname] = video_face_counts.get(vname, 0) + 1

        if cid not in person_video_stats:
            person_video_stats[cid] = {}
        if vname not in person_video_stats[cid]:
            person_video_stats[cid][vname] = {"count": 0, "min_t": r.timestamp_sec, "max_t": r.timestamp_sec}

        s = person_video_stats[cid][vname]
        s["count"] += 1
        s["min_t"] = min(s["min_t"], r.timestamp_sec)
        s["max_t"] = max(s["max_t"], r.timestamp_sec)

    persons = sorted(person_video_stats.keys())
    cross_video_persons = [cid for cid in persons if len(person_video_stats[cid]) > 1]
    single_video_persons = [cid for cid in persons if len(person_video_stats[cid]) == 1]

    # Colors
    bg_color = "#0B0F19"
    card_bg = "#151D2F"
    border_color = "#23304A"
    text_color = "#F8FAFC"
    text_muted = "#94A3B8"
    gold_highlight = "#F59E0B"
    cyan_accent = "#38BDF8"

    video_palette = [
        {"fill": "#1E3A8A", "border": "#3B82F6", "badge": "#60A5FA"},
        {"fill": "#064E3B", "border": "#10B981", "badge": "#34D399"},
        {"fill": "#581C87", "border": "#A855F7", "badge": "#C084FC"},
        {"fill": "#7C2D12", "border": "#F97316", "badge": "#FB923C"},
        {"fill": "#134E4A", "border": "#14B8A6", "badge": "#2DD4BF"},
    ]

    # Canvas dimensions
    fig, ax = plt.subplots(figsize=(16, 11), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Positions layout:
    # If 2 videos: Video 1 on left (x=0.2), Video 2 on right (x=0.8)
    # Cross-video persons placed in center column (x=0.5)!
    # Single-video persons placed outside (x=0.08 and x=0.92) or around their respective video.
    n_vids = len(video_names)
    vid_pos: Dict[str, Tuple[float, float]] = {}
    person_pos: Dict[int, Tuple[float, float]] = {}

    if n_vids == 1:
        vid_pos[video_names[0]] = (0.5, 0.5)
        # Circle layout for persons
        for i, cid in enumerate(persons):
            angle = 2 * math.pi * i / max(1, len(persons))
            person_pos[cid] = (0.5 + 0.35 * math.cos(angle), 0.5 + 0.35 * math.sin(angle))
    elif n_vids == 2:
        v1, v2 = video_names[0], video_names[1]
        vid_pos[v1] = (0.26, 0.64)
        vid_pos[v2] = (0.74, 0.64)

        # 1. Place cross-video persons in center lower
        n_cross = len(cross_video_persons)
        if n_cross > 0:
            x_start = 0.50 - (n_cross - 1) * 0.12 / 2
            for i, cid in enumerate(cross_video_persons):
                person_pos[cid] = (x_start + i * 0.12, 0.28)

        # 2. Place single-video persons outside on the flanks
        v1_singles = [cid for cid in single_video_persons if v1 in person_video_stats[cid]]
        v2_singles = [cid for cid in single_video_persons if v2 in person_video_stats[cid]]

        for i, cid in enumerate(v1_singles):
            y_pos = 0.64 + (i - (len(v1_singles) - 1) / 2) * 0.22 if len(v1_singles) > 1 else 0.40
            person_pos[cid] = (0.09, y_pos)

        for i, cid in enumerate(v2_singles):
            y_pos = 0.64 + (i - (len(v2_singles) - 1) / 2) * 0.22 if len(v2_singles) > 1 else 0.40
            person_pos[cid] = (0.91, y_pos)
    else:
        # Radial / Polygon layout for >= 3 videos
        radius_v = 0.28
        center_x, center_y = 0.5, 0.5
        for i, v in enumerate(video_names):
            angle = 2 * math.pi * i / n_vids - math.pi / 2
            vid_pos[v] = (center_x + radius_v * math.cos(angle), center_y + radius_v * math.sin(angle))

        # Position persons
        for cid in persons:
            connected_vids = list(person_video_stats[cid].keys())
            if len(connected_vids) > 1:
                # Place near center of gravity of connected videos
                gx = sum(vid_pos[v][0] for v in connected_vids) / len(connected_vids)
                gy = sum(vid_pos[v][1] for v in connected_vids) / len(connected_vids)
                person_pos[cid] = (gx, gy)
            else:
                # Place outside the single video
                v = connected_vids[0]
                vx, vy = vid_pos[v]
                dx, dy = vx - center_x, vy - center_y
                norm = math.hypot(dx, dy) or 1
                person_pos[cid] = (vx + (dx / norm) * 0.16, vy + (dy / norm) * 0.16)

    # 3. Draw Connecting Edges (Person <-> Video)
    for cid, vids in person_video_stats.items():
        if cid not in person_pos:
            continue
        px, py = person_pos[cid]
        is_cross = len(vids) > 1

        for vname, stat in vids.items():
            if vname not in vid_pos:
                continue
            vx, vy = vid_pos[vname]
            count = stat["count"]
            t_min = stat["min_t"]
            t_max = stat["max_t"]

            # Edge style
            edge_color = gold_highlight if is_cross else cyan_accent
            alpha = 0.85 if is_cross else 0.55
            lw = min(6.0, max(1.8, count * 0.5))

            # Draw curved bezier connection
            rad = 0.10 if px < vx else -0.10
            if is_cross:
                rad = 0.08 if vx < px else -0.08
            arrow = patches.FancyArrowPatch(
                (vx, vy),
                (px, py),
                connectionstyle=f"arc3,rad={rad}",
                color=edge_color,
                linewidth=lw,
                alpha=alpha,
                linestyle="solid" if is_cross else "--",
                zorder=2,
            )
            ax.add_patch(arrow)

            # Draw edge badge text positioned comfortably along the curve (towards the person)
            mx = 0.36 * vx + 0.64 * px
            my = 0.36 * vy + 0.64 * py + (0.025 if is_cross else -0.025)
            edge_lbl = f"{count}x ({t_min:.1f}s-{t_max:.1f}s)" if t_max > t_min else f"{count}x ({t_min:.1f}s)"
            ax.text(
                mx,
                my,
                edge_lbl,
                color="#E2E8F0",
                fontsize=8,
                fontweight="semibold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0F172A", ec=edge_color, lw=1, alpha=0.9),
                zorder=3,
            )

    # 4. Draw Video Nodes (Cards)
    for i, vname in enumerate(video_names):
        vx, vy = vid_pos[vname]
        col = video_palette[i % len(video_palette)]
        total_faces = video_face_counts.get(vname, 0)
        n_unique_persons = sum(1 for cid in persons if vname in person_video_stats[cid])

        # Video Card Box
        card_w, card_h = 0.18, 0.11
        rect = patches.FancyBboxPatch(
            (vx - card_w / 2, vy - card_h / 2),
            card_w,
            card_h,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=col["fill"],
            edgecolor=col["border"],
            linewidth=2.5,
            zorder=4,
        )
        ax.add_patch(rect)

        # Video Title and stats inside card
        display_vname = vname if len(vname) <= 22 else vname[:20] + "..."
        ax.text(
            vx,
            vy + 0.025,
            f"VIDEO: {display_vname}",
            color="#FFFFFF",
            fontsize=10.0,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            vx,
            vy - 0.005,
            f"{n_unique_persons} personer funnet",
            color=col["badge"],
            fontsize=8.5,
            fontweight="semibold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            vx,
            vy - 0.030,
            f"Totalt: {total_faces} ansikter",
            color="#CBD5E1",
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
        )

    # 5. Draw Person Nodes with Avatar Images
    for cid in persons:
        if cid not in person_pos:
            continue
        px, py = person_pos[cid]
        is_cross = len(person_video_stats[cid]) > 1
        ring_col = (245, 158, 11) if is_cross else (56, 189, 248)

        # Create circular portrait thumbnail
        if cid in rep_faces and rep_faces[cid] is not None:
            thumb_pil = make_circular_thumbnail(
                rep_faces[cid],
                size=110,
                ring_color=ring_col,
                ring_width=6 if is_cross else 4,
            )
            # Embed image in matplotlib
            imbox = OffsetImage(np.array(thumb_pil), zoom=0.55)
            ab = AnnotationBbox(imbox, (px, py), frameon=False, zorder=6)
            ax.add_artist(ab)
        else:
            # Fallback circle if image missing
            circle = plt.Circle((px, py), 0.03, color=card_bg, ec=cyan_accent, lw=2, zorder=6)
            ax.add_patch(circle)

        # Person Label Below Avatar
        label_text = f"Person #{cid:03d}"
        if is_cross:
            label_text += " [Kryss-video]"

        lbl_color = gold_highlight if is_cross else "#FFFFFF"
        ax.text(
            px,
            py - 0.046,
            label_text,
            color=lbl_color,
            fontsize=8.5,
            fontweight="bold" if is_cross else "normal",
            ha="center",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="#0B0F19",
                ec=gold_highlight if is_cross else border_color,
                lw=1.2 if is_cross else 0.8,
            ),
            zorder=7,
        )

    # 6. Header Titles and Subtitles
    ax.text(
        0.5,
        0.96,
        "TANKEDIAGRAM: KOBLINGER MELLOM VIDEOER & PERSONER",
        color="#FFFFFF",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="top",
    )
    subtitle = (
        f"Gjenkjenning og koblinger pa tvers av {len(video_names)} video(er) | "
        f"{len(persons)} unike personer oppdaget | "
        f"{len(cross_video_persons)} kryss-video person(er)"
    )
    ax.text(
        0.5,
        0.925,
        subtitle,
        color=text_muted,
        fontsize=10.0,
        ha="center",
        va="top",
    )

    # 7. Summary / Legend Box in bottom left
    legend_text = (
        "TEGNFORKLARING:\n"
        "- Bla/Gronne kort = Kildevideoer\n"
        "- Sirkler med foto = Oppdagede person-identiteter\n"
        "- Gullring/Stiplet = Kryss-video person (funnet i flere videoer)\n"
        "- Kantetiketter = Antall opptredener og tidsintervall"
    )
    ax.text(
        0.03,
        0.04,
        legend_text,
        color="#CBD5E1",
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc=card_bg, ec=border_color, lw=1.2, alpha=0.95),
        zorder=8,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Static Mind Map Diagram saved to: {output_path}")
    return output_path


def generate_interactive_dashboard(
    face_records: List[Any],
    output_dir: str,
    stats: Dict[str, Any],
    rep_face_rel_paths: Dict[int, str],
    video_paths: List[str],
) -> str:
    """Generates a complete, interactive single-page HTML application (index.html).
    
    Contains:
    1. Dynamic Interactive Mind Map (Canvas physics graph with avatar thumbnails).
    2. Integrated Video Player with one-click jump-to-timestamp.
    3. Filterable Face Gallery & Cross-Video Matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "index.html")

    # Serialize face data cleanly for client-side JS
    client_faces = []
    client_videos = {}

    for vp in video_paths:
        vname = os.path.splitext(os.path.basename(vp))[0]
        # Calculate relative path from output_dir to source video
        rel_vpath = os.path.relpath(vp, output_dir).replace("\\", "/")
        client_videos[vname] = {
            "name": vname,
            "path": rel_vpath,
            "abs_path": os.path.abspath(vp),
            "face_count": 0,
        }

    for r in face_records:
        cid = r.cluster_id
        hms = format_seconds_to_hms(r.timestamp_sec)
        vname = r.video_name
        if vname in client_videos:
            client_videos[vname]["face_count"] += 1

        rel_img = r.saved_path.replace("\\", "/") if r.saved_path else ""
        rel_jump = getattr(r, "jump_file_path", "")
        if rel_jump:
            rel_jump = rel_jump.replace("\\", "/")

        client_faces.append({
            "person_id": cid,
            "person_label": f"person_{cid:03d}" if cid != -1 else "unclustered",
            "video_name": vname,
            "video_path": client_videos.get(vname, {}).get("path", ""),
            "timestamp_sec": round(r.timestamp_sec, 3),
            "timestamp_hms": hms,
            "frame_idx": r.frame_idx,
            "confidence": round(r.confidence, 4),
            "image_path": rel_img,
            "jump_path": rel_jump,
            "bbox": r.bbox,
        })

    # Prepare person summary
    persons_summary = {}
    for f in client_faces:
        cid = f["person_id"]
        if cid == -1:
            continue
        if cid not in persons_summary:
            persons_summary[cid] = {
                "id": cid,
                "label": f"person_{cid:03d}",
                "rep_img": rep_face_rel_paths.get(cid, "").replace("\\", "/"),
                "total_count": 0,
                "videos": {},
            }
        p = persons_summary[cid]
        p["total_count"] += 1
        vn = f["video_name"]
        if vn not in p["videos"]:
            p["videos"][vn] = {"count": 0, "first_sec": f["timestamp_sec"], "last_sec": f["timestamp_sec"]}
        p["videos"][vn]["count"] += 1
        p["videos"][vn]["first_sec"] = min(p["videos"][vn]["first_sec"], f["timestamp_sec"])
        p["videos"][vn]["last_sec"] = max(p["videos"][vn]["last_sec"], f["timestamp_sec"])

    # Count cross video persons
    cross_video_count = sum(1 for p in persons_summary.values() if len(p["videos"]) > 1)

    html_code = f"""<!DOCTYPE html>
<html lang="no">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Video Ansiktsklynging & Tankediagram Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #0a0e17;
      --bg-card: #121826;
      --bg-card-hover: #1a2236;
      --border: #1e293b;
      --border-light: #334155;
      --accent: #38bdf8;
      --accent-rgb: 56, 189, 248;
      --gold: #f59e0b;
      --green: #10b981;
      --purple: #a855f7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg-dark);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }}
    header {{
      background: rgba(18, 24, 38, 0.8);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 16px 28px;
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .brand-icon {{
      font-size: 1.8rem;
    }}
    .brand-text h1 {{
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.3px;
    }}
    .brand-text p {{
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .stats-bar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .stat-pill {{
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.82rem;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .stat-pill strong {{
      color: var(--accent);
      font-weight: 700;
    }}
    .stat-pill.gold strong {{ color: var(--gold); }}
    .nav-tabs {{
      display: flex;
      gap: 8px;
      background: var(--bg-dark);
      padding: 6px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }}
    .nav-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.88rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .nav-btn:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.05);
    }}
    .nav-btn.active {{
      background: var(--accent);
      color: #0b0f19;
      box-shadow: 0 0 16px rgba(var(--accent-rgb), 0.35);
    }}
    main {{
      padding: 24px 28px;
      max-width: 1600px;
      margin: 0 auto;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}

    /* MIND MAP TAB */
    .mindmap-wrapper {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
      height: 780px;
    }}
    @media (max-width: 1024px) {{
      .mindmap-wrapper {{ grid-template-columns: 1fr; height: auto; }}
    }}
    .canvas-container {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      position: relative;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .canvas-toolbar {{
      position: absolute;
      top: 14px;
      left: 14px;
      z-index: 10;
      display: flex;
      gap: 8px;
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(8px);
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .tool-btn {{
      background: #1e293b;
      border: 1px solid #334155;
      color: #fff;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .tool-btn:hover {{ background: #334155; }}
    #mindmapCanvas {{
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    #mindmapCanvas:active {{ cursor: grabbing; }}
    .sidebar-panel {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      overflow-y: auto;
      max-height: 780px;
    }}
    .sidebar-title {{
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .selected-node-details {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .avatar-preview {{
      width: 100%;
      max-height: 200px;
      object-fit: cover;
      border-radius: 12px;
      border: 2px solid var(--accent);
    }}
    .appearance-item {{
      background: rgba(30, 41, 59, 0.4);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }}
    .jump-btn-small {{
      background: #0284c7;
      color: #fff;
      border: none;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .jump-btn-small:hover {{ background: #0369a1; }}

    /* VIDEO PLAYER TAB */
    .player-layout {{
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 24px;
    }}
    @media (max-width: 1100px) {{
      .player-layout {{ grid-template-columns: 1fr; }}
    }}
    .main-video-box {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    video#interactivePlayer {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      max-height: 540px;
    }}
    .player-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .select-dropdown {{
      background: #1e293b;
      color: #fff;
      border: 1px solid #334155;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.9rem;
      outline: none;
    }}
    .timeline-list {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      overflow-y: auto;
      max-height: 700px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .timeline-card {{
      display: flex;
      align-items: center;
      gap: 12px;
      background: #151d2f;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .timeline-card:hover {{
      background: var(--bg-card-hover);
      border-color: var(--accent);
      transform: translateX(3px);
    }}
    .timeline-thumb {{
      width: 54px;
      height: 54px;
      border-radius: 8px;
      object-fit: cover;
      border: 1px solid #334155;
    }}
    .timeline-info {{ flex: 1; }}
    .timeline-time {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      font-weight: 700;
      font-size: 0.88rem;
    }}
    .timeline-meta {{
      font-size: 0.78rem;
      color: var(--text-muted);
    }}

    /* GALLERY TAB */
    .filter-bar {{
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 20px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 14px 20px;
      border-radius: 12px;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 16px;
    }}
    .face-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: all 0.2s ease;
      position: relative;
    }}
    .face-card:hover {{
      border-color: var(--accent);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      transform: translateY(-2px);
    }}
    .face-img {{
      width: 100%;
      height: 160px;
      object-fit: cover;
      border-radius: 8px;
      background: #000;
    }}
    .cross-badge {{
      position: absolute;
      top: 18px;
      right: 18px;
      background: var(--gold);
      color: #000;
      font-size: 0.7rem;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 4px;
    }}
    .face-card-details {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .face-card-details strong {{ color: #fff; }}
    .time-badge {{
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent);
      font-weight: 600;
    }}

    /* MATRIX TAB */
    .matrix-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    .matrix-table th, .matrix-table td {{
      padding: 14px 18px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    .matrix-table th {{
      background: #151d2f;
      color: #fff;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .matrix-table tr:hover {{
      background: rgba(255, 255, 255, 0.02);
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-text">
        <h1>Video Ansiktsklynging & Tankediagram</h1>
        <p>Kryss-video analyse, tidsreferanser og nettverkskoblinger</p>
      </div>
    </div>

    <div class="stats-bar">
      <div class="stat-pill">Videoer: <strong>{len(client_videos)}</strong></div>
      <div class="stat-pill">Unike Personer: <strong>{len(persons_summary)}</strong></div>
      <div class="stat-pill gold">Kryss-video Personer: <strong>{cross_video_count}</strong></div>
      <div class="stat-pill">Ansikter: <strong>{len(client_faces)}</strong></div>
      <div class="stat-pill">Metodikk: <strong>{stats.get("method", "dbscan").upper()}</strong></div>
    </div>

    <nav class="nav-tabs">
      <button class="nav-btn active" data-tab="tab-mindmap" onclick="switchTab('tab-mindmap')">Tankediagram</button>
      <button class="nav-btn" data-tab="tab-player" onclick="switchTab('tab-player')">Videospiller & Tidshopp</button>
      <button class="nav-btn" data-tab="tab-gallery" onclick="switchTab('tab-gallery')">Ansiktsgalleri</button>
      <button class="nav-btn" data-tab="tab-matrix" onclick="switchTab('tab-matrix')">Kryss-video Matrise</button>
    </nav>
  </header>

  <main>
    <!-- 1. MIND MAP TAB -->
    <section id="tab-mindmap" class="tab-content active">
      <div class="mindmap-wrapper">
        <div class="canvas-container">
          <div class="canvas-toolbar">
            <button class="tool-btn" onclick="resetCanvasView()">Tilbakestill visning</button>
            <button class="tool-btn" onclick="zoomCanvas(1.2)">Zoom +</button>
            <button class="tool-btn" onclick="zoomCanvas(0.8)">Zoom -</button>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">Dra noder for å flytte • Klikk node for detaljer</span>
          </div>
          <canvas id="mindmapCanvas"></canvas>
        </div>

        <div class="sidebar-panel" id="mindmapSidebar">
          <div class="sidebar-title">
            <span>Valgt Node / Identitet</span>
          </div>
          <div id="nodeDetailsPlaceholder" style="color: #94a3b8; font-size: 0.9rem; text-align: center; margin-top: 60px;">
            Klikk på en person- eller videonode i tankediagrammet for å se alle tidsstempler og opptredener.
          </div>
          <div id="nodeDetailsContent" class="selected-node-details" style="display: none;"></div>
        </div>
      </div>
    </section>

    <!-- 2. VIDEO PLAYER TAB -->
    <section id="tab-player" class="tab-content">
      <div class="player-layout">
        <div class="main-video-box">
          <div class="player-toolbar">
            <div style="display: flex; align-items: center; gap: 10px;">
              <label for="videoSelect" style="font-weight: 600; font-size: 0.9rem;">Velg Video:</label>
              <select id="videoSelect" class="select-dropdown" onchange="onVideoSelectChange()">
                {"".join(f'<option value="{v["name"]}">{v["name"]} ({v["face_count"]} ansikter)</option>' for v in client_videos.values())}
              </select>
            </div>
            <div id="currentTimestampBadge" class="stat-pill" style="font-family: 'JetBrains Mono', monospace;">
              Tid: <strong id="playerTimeDisplay">00:00.00</strong>
            </div>
          </div>

          <video id="interactivePlayer" controls preload="metadata">
            <source id="videoPlayerSource" src="" type="video/mp4">
            Nettleseren støtter ikke video.
          </video>

          <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <button class="tool-btn" onclick="stepPlayer(-1)">-1s</button>
            <button class="tool-btn" onclick="stepPlayer(-0.1)">-0.1s</button>
            <button class="tool-btn" onclick="stepPlayer(0.1)">+0.1s</button>
            <button class="tool-btn" onclick="stepPlayer(1)">+1s</button>
            <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">Tips: Klikk på et ansiktskort til høyre for å hoppe direkte til tidspunktet!</span>
          </div>
        </div>

        <div class="timeline-list">
          <div class="sidebar-title">
            <span>Ansiktsopptredener i valgt video</span>
          </div>
          <div id="timelineCardsContainer"></div>
        </div>
      </div>
    </section>

    <!-- 3. GALLERY TAB -->
    <section id="tab-gallery" class="tab-content">
      <div class="filter-bar">
        <label style="font-size: 0.9rem; font-weight: 600;">Filtrer etter person:</label>
        <select id="personFilter" class="select-dropdown" onchange="renderGallery()">
          <option value="ALL">Alle Personer</option>
          {"".join(f'<option value="{p["id"]}">{p["label"]} ({p["total_count"]} bilder)</option>' for p in persons_summary.values())}
          <option value="-1">Uklassifisert (Støy)</option>
        </select>

        <label style="font-size: 0.9rem; font-weight: 600; margin-left: 12px;">Filtrer etter video:</label>
        <select id="videoFilter" class="select-dropdown" onchange="renderGallery()">
          <option value="ALL">Alle Videoer</option>
          {"".join(f'<option value="{v["name"]}">{v["name"]}</option>' for v in client_videos.values())}
        </select>

        <label style="margin-left: 16px; font-size: 0.88rem; cursor: pointer; display: flex; align-items: center; gap: 6px;">
          <input type="checkbox" id="crossOnlyCheck" onchange="renderGallery()">
          Vis kun kryss-video personer
        </label>
      </div>

      <div class="gallery-grid" id="galleryContainer"></div>
    </section>

    <!-- 4. MATRIX TAB -->
    <section id="tab-matrix" class="tab-content">
      <div style="background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; padding: 24px;">
        <h2 style="font-size: 1.2rem; margin-bottom: 6px;">Kryss-video Forekomstmatrise</h2>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 20px;">
          Tabellen viser hvilke videoer hver enkelt person opptrer i, totalt antall ansikter, og tidsintervall.
        </p>

        <table class="matrix-table">
          <thead>
            <tr>
              <th>Identitet</th>
              <th>Type</th>
              {"".join(f'<th>Video: {vname}</th>' for vname in client_videos.keys())}
              <th>Totalt Antall</th>
              <th>Handling</th>
            </tr>
          </thead>
          <tbody>
            {"".join(f'''
            <tr>
              <td>
                <div style="display: flex; align-items: center; gap: 10px;">
                  <img src="{p["rep_img"]}" style="width: 38px; height: 38px; border-radius: 8px; object-fit: cover; border: 1px solid #334155;">
                  <strong>{p["label"]}</strong>
                </div>
              </td>
              <td>
                {"<span style='background: #f59e0b; color: #000; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;'>KRYSS-VIDEO</span>" if len(p["videos"]) > 1 else "<span style='color: #94a3b8; font-size: 0.8rem;'>Enkeltvideo</span>"}
              </td>
              {"".join(f"<td>{'<strong>' + str(p['videos'][vname]['count']) + 'x</strong> (' + format_seconds_to_hms(p['videos'][vname]['first_sec']) + ')' if vname in p['videos'] else '<span style=\"color: #475569;\">-</span>'}</td>" for vname in client_videos.keys())}
              <td><strong>{p["total_count"]} ansikter</strong></td>
              <td>
                <button class="jump-btn-small" onclick="selectPersonInMindmap({p['id']})">Vis i Tankediagram</button>
              </td>
            </tr>
            ''' for p in persons_summary.values())}
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    // Embedded Data
    const FACES_DATA = {json.dumps(client_faces)};
    const VIDEOS_DATA = {json.dumps(client_videos)};
    const PERSONS_DATA = {json.dumps(persons_summary)};

    // Navigation Tabs
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
      const targetTab = document.getElementById(tabId);
      if (targetTab) targetTab.classList.add('active');
      const targetBtn = document.querySelector(`.nav-btn[data-tab="${{tabId}}"]`);
      if (targetBtn) targetBtn.classList.add('active');

      if (tabId === 'tab-mindmap') {{
        requestAnimationFrame(renderMindmap);
      }} else if (tabId === 'tab-player') {{
        setupPlayerTab();
      }} else if (tabId === 'tab-gallery') {{
        renderGallery();
      }}
    }}

    // ==========================================
    // INTERACTIVE MIND MAP ENGINE (CANVAS)
    // ==========================================
    const canvas = document.getElementById('mindmapCanvas');
    const ctx = canvas.getContext('2d');
    let nodes = [];
    let edges = [];
    let transform = {{ x: 0, y: 0, k: 1 }};
    let isDragging = false;
    let dragNode = null;
    let startX = 0, startY = 0;
    let selectedNode = null;
    const nodeImages = {{}};

    function initMindmap() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      // Preload avatar images
      for (const p of Object.values(PERSONS_DATA)) {{
        if (p.rep_img) {{
          const img = new Image();
          img.src = p.rep_img;
          nodeImages[p.id] = img;
        }}
      }}

      // Build Graph Nodes
      nodes = [];
      edges = [];
      const vKeys = Object.keys(VIDEOS_DATA);
      const width = rect.width;
      const height = rect.height;

      // Video Nodes
      vKeys.forEach((vname, idx) => {{
        const angle = (2 * Math.PI * idx) / vKeys.length - Math.PI / 2;
        const radius = Math.min(width, height) * 0.28;
        const vx = width / 2 + (vKeys.length === 2 ? (idx === 0 ? -180 : 180) : radius * Math.cos(angle));
        const vy = height / 2 + (vKeys.length === 2 ? 0 : radius * Math.sin(angle));

        nodes.push({{
          id: 'v_' + vname,
          type: 'video',
          name: vname,
          x: vx,
          y: vy,
          vx: 0, vy: 0,
          radius: 45,
          color: idx % 2 === 0 ? '#1e3a8a' : '#064e3b',
          borderColor: idx % 2 === 0 ? '#3b82f6' : '#10b981',
          data: VIDEOS_DATA[vname]
        }});
      }});

      // Person Nodes
      const pList = Object.values(PERSONS_DATA);
      pList.forEach((p, idx) => {{
        const isCross = Object.keys(p.videos).length > 1;
        let px, py;
        if (isCross) {{
          // Center between connected videos
          px = width / 2 + (idx % 2 === 0 ? -20 : 20);
          py = height / 2 - 120 + idx * 50;
        }} else {{
          const vname = Object.keys(p.videos)[0];
          const vNode = nodes.find(n => n.id === 'v_' + vname);
          const vIdx = vKeys.indexOf(vname);
          const angle = Math.PI * (vIdx === 0 ? 0.6 + 0.8 * (idx / pList.length) : -0.4 + 0.8 * (idx / pList.length));
          px = (vNode ? vNode.x : width / 2) + 160 * Math.cos(angle);
          py = (vNode ? vNode.y : height / 2) + 160 * Math.sin(angle);
        }}

        nodes.push({{
          id: 'p_' + p.id,
          type: 'person',
          name: p.label,
          personId: p.id,
          x: px,
          y: py,
          vx: 0, vy: 0,
          radius: 34,
          isCross: isCross,
          data: p
        }});

        // Edges
        for (const [vn, stat] of Object.entries(p.videos)) {{
          edges.push({{
            source: 'p_' + p.id,
            target: 'v_' + vn,
            count: stat.count,
            min_sec: stat.first_sec,
            max_sec: stat.last_sec,
            isCross: isCross
          }});
        }}
      }});

      transform = {{ x: 0, y: 0, k: 1 }};
      renderMindmap();
    }}

    function renderMindmap() {{
      const rect = canvas.getBoundingClientRect();
      ctx.save();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // Draw Edges
      edges.forEach(e => {{
        const s = nodes.find(n => n.id === e.source);
        const t = nodes.find(n => n.id === e.target);
        if (!s || !t) return;

        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.strokeStyle = e.isCross ? '#f59e0b' : '#38bdf8';
        ctx.lineWidth = e.isCross ? 3.5 : 2;
        ctx.setLineDash(e.isCross ? [] : [4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label on edge
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(mx - 25, my - 10, 50, 18);
        ctx.strokeStyle = e.isCross ? '#f59e0b' : '#334155';
        ctx.lineWidth = 1;
        ctx.strokeRect(mx - 25, my - 10, 50, 18);

        ctx.fillStyle = '#f8fafc';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(e.count + 'x', mx, my - 1);
      }});

      // Draw Nodes
      nodes.forEach(n => {{
        ctx.save();
        if (n.type === 'video') {{
          // Video Pill Box
          ctx.beginPath();
          ctx.roundRect(n.x - 70, n.y - 30, 140, 60, 12);
          ctx.fillStyle = n.color;
          ctx.fill();
          ctx.strokeStyle = n.borderColor;
          ctx.lineWidth = 2.5;
          ctx.stroke();

          ctx.fillStyle = '#fff';
          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(n.name, n.x, n.y - 6);

          ctx.fillStyle = '#94a3b8';
          ctx.font = '10px Inter, sans-serif';
          ctx.fillText(n.data.face_count + ' ansikter funnet', n.x, n.y + 12);
        }} else {{
          // Person Avatar Node
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, 0, 2 * Math.PI);
          ctx.fillStyle = '#1e293b';
          ctx.fill();

          // Clip image
          const img = nodeImages[n.personId];
          if (img && img.complete) {{
            ctx.save();
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.radius - 2, 0, 2 * Math.PI);
            ctx.clip();
            ctx.drawImage(img, n.x - n.radius, n.y - n.radius, n.radius * 2, n.radius * 2);
            ctx.restore();
          }}

          // Border Ring
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.radius, 0, 2 * Math.PI);
          ctx.strokeStyle = n.isCross ? '#f59e0b' : '#38bdf8';
          ctx.lineWidth = n.isCross ? 4.5 : 2.5;
          ctx.stroke();

          // Label
          ctx.fillStyle = n.isCross ? '#f59e0b' : '#fff';
          ctx.font = (n.isCross ? 'bold 11px' : '10px') + ' Inter, sans-serif';
          ctx.textAlign = 'center';
          const tag = n.isCross ? ' [Kryss]' : '';
          ctx.fillText(n.name + tag, n.x, n.y + n.radius + 16);
        }}
        ctx.restore();
      }});

      ctx.restore();
    }}

    // Canvas Interactions (Drag, Zoom, Click)
    canvas.addEventListener('mousedown', e => {{
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left - transform.x) / transform.k;
      const my = (e.clientY - rect.top - transform.y) / transform.k;

      // Find node under mouse
      dragNode = nodes.find(n => Math.hypot(n.x - mx, n.y - my) <= (n.radius || 45));
      if (dragNode) {{
        selectNode(dragNode);
      }} else {{
        isDragging = true;
        startX = e.clientX - transform.x;
        startY = e.clientY - transform.y;
      }}
    }});

    window.addEventListener('mousemove', e => {{
      if (dragNode) {{
        const rect = canvas.getBoundingClientRect();
        dragNode.x = (e.clientX - rect.left - transform.x) / transform.k;
        dragNode.y = (e.clientY - rect.top - transform.y) / transform.k;
        renderMindmap();
      }} else if (isDragging) {{
        transform.x = e.clientX - startX;
        transform.y = e.clientY - startY;
        renderMindmap();
      }}
    }});

    window.addEventListener('mouseup', () => {{
      dragNode = null;
      isDragging = false;
    }});

    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      zoomCanvas(zoomFactor);
    }});

    function zoomCanvas(factor) {{
      transform.k = Math.max(0.3, Math.min(3.0, transform.k * factor));
      renderMindmap();
    }}

    function resetCanvasView() {{
      transform = {{ x: 0, y: 0, k: 1 }};
      renderMindmap();
    }}

    function selectNode(n) {{
      selectedNode = n;
      document.getElementById('nodeDetailsPlaceholder').style.display = 'none';
      const container = document.getElementById('nodeDetailsContent');
      container.style.display = 'flex';

      if (n.type === 'person') {{
        const p = n.data;
        const appearances = FACES_DATA.filter(f => f.person_id === p.id);
        const isCross = Object.keys(p.videos).length > 1;

        let html = `
          <img src="${{p.rep_img}}" class="avatar-preview" alt="${{p.label}}">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="font-size: 1.1rem; color: #fff;">${{p.label}}</h2>
            ${{isCross ? '<span style="background: #f59e0b; color: #000; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;">KRYSS-VIDEO</span>' : ''}}
          </div>
          <p style="font-size: 0.85rem; color: #94a3b8;">Funnet i <strong>${{Object.keys(p.videos).length}}</strong> video(er) med totalt <strong>${{p.total_count}}</strong> ansikter.</p>
          <div style="font-size: 0.9rem; font-weight: 600; margin-top: 6px;">Tidsstempler & Hurtighopp:</div>
        `;

        appearances.slice(0, 15).forEach(f => {{
          html += `
            <div class="appearance-item">
              <div>
                <strong style="color: #38bdf8;">${{f.video_name}}</strong>
                <div style="color: #94a3b8; font-size: 0.8rem;">${{f.timestamp_hms}} (${{f.timestamp_sec.toFixed(2)}}s)</div>
              </div>
              <button class="jump-btn-small" onclick="jumpToFromSidebar('${{f.video_name}}', ${{f.timestamp_sec}})">Hopp hit</button>
            </div>
          `;
        }});

        container.innerHTML = html;
      }} else {{
        // Video Node selected
        const v = n.data;
        const appearances = FACES_DATA.filter(f => f.video_name === n.name);
        const uniquePersonsInVid = [...new Set(appearances.map(f => f.person_label))];

        let html = `
          <h2 style="font-size: 1.1rem; color: #fff;">Video: ${{n.name}}</h2>
          <p style="font-size: 0.85rem; color: #94a3b8;">${{uniquePersonsInVid.length}} unike personer, ${{v.face_count}} ansikter totalt.</p>
          <div style="margin-top: 8px;">
            <button class="tool-btn" style="width: 100%; padding: 8px;" onclick="loadVideoInPlayer('${{n.name}}')">Åpne Video i Spiller</button>
          </div>
          <div style="font-size: 0.85rem; font-weight: 600; margin-top: 10px;">Personer i denne videoen:</div>
        `;

        uniquePersonsInVid.forEach(pLbl => {{
          html += `<div class="stat-pill" style="margin-top: 4px;">${{pLbl}}</div>`;
        }});

        container.innerHTML = html;
      }}
    }}

    function selectPersonInMindmap(personId) {{
      switchTab('tab-mindmap');
      const target = nodes.find(n => n.personId === personId);
      if (target) {{
        selectNode(target);
      }}
    }}

    // ==========================================
    // VIDEO PLAYER TAB
    // ==========================================
    const player = document.getElementById('interactivePlayer');
    const playerTimeDisplay = document.getElementById('playerTimeDisplay');

    function setupPlayerTab() {{
      const select = document.getElementById('videoSelect');
      if (select.value) {{
        loadVideoInPlayer(select.value);
      }}
    }}

    function onVideoSelectChange() {{
      const select = document.getElementById('videoSelect');
      loadVideoInPlayer(select.value);
    }}

    function loadVideoInPlayer(videoName, jumpSec = null) {{
      const v = VIDEOS_DATA[videoName];
      if (!v) return;

      const select = document.getElementById('videoSelect');
      if (select && select.value !== videoName) select.value = videoName;

      const currentSrc = player.getAttribute('data-loaded-video');
      if (currentSrc !== videoName) {{
        player.setAttribute('data-loaded-video', videoName);
        player.src = v.path;
        player.load();
      }}

      const doSeek = () => {{
        if (jumpSec !== null) {{
          player.currentTime = jumpSec;
          player.play().catch(e => console.log('Autoplay deferred:', e));
        }}
      }};

      if (player.readyState >= 1) {{
        doSeek();
      }} else {{
        player.onloadedmetadata = () => {{
          doSeek();
          player.onloadedmetadata = null;
        }};
      }}

      // Render timeline list for this video
      const container = document.getElementById('timelineCardsContainer');
      const vidFaces = FACES_DATA.filter(f => f.video_name === videoName);
      vidFaces.sort((a, b) => a.timestamp_sec - b.timestamp_sec);

      let cardsHtml = '';
      vidFaces.forEach(f => {{
        cardsHtml += `
          <div class="timeline-card" onclick="seekPlayerTo(${{f.timestamp_sec}})">
            <img class="timeline-thumb" src="${{f.image_path}}" alt="Ansikt">
            <div class="timeline-info">
              <div class="timeline-time">${{f.timestamp_hms}}</div>
              <div class="timeline-meta">${{f.person_label}} • Ramme #${{f.frame_idx}}</div>
            </div>
            <button class="jump-btn-small">Hopp</button>
          </div>
        `;
      }});
      container.innerHTML = cardsHtml || '<div style="color: #94a3b8; text-align: center; padding: 20px;">Ingen ansikter i denne videoen.</div>';
    }}

    function seekPlayerTo(seconds) {{
      player.currentTime = seconds;
      player.play().catch(e => {{}});
    }}

    function stepPlayer(delta) {{
      player.currentTime = Math.max(0, player.currentTime + delta);
    }}

    function jumpToFromSidebar(videoName, seconds) {{
      switchTab('tab-player');
      loadVideoInPlayer(videoName, seconds);
    }}

    player.addEventListener('timeupdate', () => {{
      const s = player.currentTime;
      const m = Math.floor(s / 60);
      const sec = (s % 60).toFixed(2);
      playerTimeDisplay.innerText = String(m).padStart(2, '0') + ':' + String(sec).padStart(5, '0');
    }});

    // ==========================================
    // GALLERY TAB
    // ==========================================
    function renderGallery() {{
      const personVal = document.getElementById('personFilter').value;
      const videoVal = document.getElementById('videoFilter').value;
      const crossOnly = document.getElementById('crossOnlyCheck').checked;

      const container = document.getElementById('galleryContainer');
      let filtered = FACES_DATA;

      if (personVal !== 'ALL') {{
        filtered = filtered.filter(f => String(f.person_id) === personVal);
      }}
      if (videoVal !== 'ALL') {{
        filtered = filtered.filter(f => f.video_name === videoVal);
      }}
      if (crossOnly) {{
        filtered = filtered.filter(f => {{
          const p = PERSONS_DATA[f.person_id];
          return p && Object.keys(p.videos).length > 1;
        }});
      }}

      let html = '';
      filtered.forEach(f => {{
        const p = PERSONS_DATA[f.person_id];
        const isCross = p && Object.keys(p.videos).length > 1;

        html += `
          <div class="face-card">
            ${{isCross ? '<div class="cross-badge">KRYSS</div>' : ''}}
            <img class="face-img" src="${{f.image_path}}" alt="Ansikt">
            <div class="face-card-details">
              <strong>${{f.person_label}}</strong>
              <div style="color: #cbd5e1;">Video: ${{f.video_name}}</div>
              <div class="time-badge">${{f.timestamp_hms}}</div>
              <div>Ramme: #${{f.frame_idx}} (${{(f.confidence * 100).toFixed(1)}}%)</div>
            </div>
            <div style="display: flex; gap: 6px; margin-top: 6px;">
              <button class="jump-btn-small" style="flex: 1;" onclick="jumpToFromSidebar('${{f.video_name}}', ${{f.timestamp_sec}})">Hopp til video</button>
              ${{f.jump_path ? `<a href="${{f.jump_path}}" target="_blank" class="tool-btn" style="text-decoration: none;" title="Åpne direkte hoppfil">Fil</a>` : ''}}
            </div>
          </div>
        `;
      }});

      container.innerHTML = html || '<div style="color: #94a3b8; grid-column: 1/-1; text-align: center; padding: 40px;">Ingen ansikter matcher filteret.</div>';
    }}

    // Initialize on window load
    window.addEventListener('DOMContentLoaded', () => {{
      initMindmap();
      renderGallery();
    }});
  </script>
</body>
</html>
"""
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html_code)

    # Also save a standalone mindmap copy as mindmap_diagram.html for convenience
    mindmap_html_file = os.path.join(output_dir, "mindmap_diagram.html")
    with open(mindmap_html_file, "w", encoding="utf-8") as f:
        f.write(html_code)

    print(f"[+] Interactive Dashboard & Mind Map saved to: {index_file}")
    return index_file
