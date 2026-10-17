#!/usr/bin/env bash
# ==============================================================================
#  Video Face Recognition, Cross-Video Clustering & Mind Map Runner (Unix/macOS)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="cross_video_output"
FRAME_INTERVAL=5
METHOD="dbscan"
NO_BROWSER=0
VIDEO_DIR=""
VIDEOS=()
DEMO_MODE=0

# 1. Detect Python & Virtual Environment
PYTHON_BIN="python3"
if [ -f "./.venv/bin/python" ]; then
    PYTHON_BIN="./.venv/bin/python"
    echo "[*] Bruker virtuelt miljø: $PYTHON_BIN"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "Feil: Python ble ikke funnet på systemet. Vennligst installer Python 3."
    exit 1
fi

# 2. Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo)
            DEMO_MODE=1
            shift
            ;;
        --video-dir|--dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --frame-interval|-i)
            FRAME_INTERVAL="$2"
            shift 2
            ;;
        --method|-m)
            METHOD="$2"
            shift 2
            ;;
        --no-browser)
            NO_BROWSER=1
            shift
            ;;
        --help|-h)
            echo "Bruk: ./run_clusterer.sh [valg] [videofiler...]"
            echo ""
            echo "Valg:"
            echo "  --demo               Kjør analyse på det inkluderte sample_videos-settet"
            echo "  --dir <mappe>        Skann alle videoer i en mappe"
            echo "  --output <mappe>     Mappe for resultater (standard: cross_video_output)"
            echo "  --frame-interval <N> Sample hver N-te frame (standard: 5)"
            echo "  --method <navn>      Klyngealgoritme: dbscan eller agglomerative"
            echo "  --no-browser         Ikke åpne nettleseren automatisk etter analyse"
            echo "  -h, --help           Vis denne hjelpen"
            exit 0
            ;;
        *)
            VIDEOS+=("$1")
            shift
            ;;
    esac
done

echo "======================================================================"
echo "     VIDEO ANSIKTSKLUDERING & TANKEDIAGRAM - UNIX RUNNER"
echo "======================================================================"

# 3. Interactive prompt if no arguments provided
if [ $DEMO_MODE -eq 1 ]; then
    VIDEOS=("sample_videos/video_intervju_del1.mp4" "sample_videos/video_intervju_del2.mp4")
elif [ ${#VIDEOS[@]} -eq 0 ] && [ -z "$VIDEO_DIR" ]; then
    echo ""
    echo "Velg hva du onsker a gjore:"
    echo "  [1] Demo: Kjor kryss-video analyse pa sample_videos/ (2 videoer med felles person)"
    echo "  [2] Skann en hel mappe med videoer"
    echo "  [3] Velg videofiler manuelt"
    echo "  [4] Generer testvideoer pa nytt (generate_sample_video.py)"
    echo "  [Q] Avslutt"
    read -rp "Skriv valg [1-4, Q]: " CHOICE

    case "$CHOICE" in
        1)
            if [ ! -f "sample_videos/video_intervju_del1.mp4" ]; then
                echo "[*] Genererer testvideoer forst..."
                $PYTHON_BIN generate_sample_video.py
            fi
            VIDEOS=("sample_videos/video_intervju_del1.mp4" "sample_videos/video_intervju_del2.mp4")
            ;;
        2)
            read -rp "Skriv inn banen til videomappen: " VIDEO_DIR
            if [ ! -d "$VIDEO_DIR" ]; then
                echo "Feil: Mappen '$VIDEO_DIR' finnes ikke."
                exit 1
            fi
            ;;
        3)
            read -rp "Skriv inn videofiler atskilt med mellomrom: " -a VIDEOS
            ;;
        4)
            echo "[*] Genererer testvideoer..."
            $PYTHON_BIN generate_sample_video.py
            VIDEOS=("sample_videos/video_intervju_del1.mp4" "sample_videos/video_intervju_del2.mp4")
            ;;
        [Qq]*)
            echo "Avbrutt."
            exit 0
            ;;
        *)
            echo "Ugyldig valg. Avbryter."
            exit 1
            ;;
    esac
fi

# 4. Construct arguments for main.py
MAIN_ARGS=("--output" "$OUTPUT_DIR" "--frame-interval" "$FRAME_INTERVAL" "--method" "$METHOD")

if [ -n "$VIDEO_DIR" ]; then
    MAIN_ARGS+=("--video-dir" "$VIDEO_DIR")
fi

if [ ${#VIDEOS[@]} -gt 0 ]; then
    MAIN_ARGS+=("--videos" "${VIDEOS[@]}")
fi

echo ""
echo "[*] Kjorer analyse: $PYTHON_BIN main.py ${MAIN_ARGS[*]}"
echo "----------------------------------------------------------------------"

$PYTHON_BIN main.py "${MAIN_ARGS[@]}"

INDEX_PATH="$OUTPUT_DIR/index.html"
if [ -f "$INDEX_PATH" ]; then
    echo ""
    echo "[+] Fullfort! Resultatene er lagret i: $OUTPUT_DIR"
    echo "    - Interaktivt dashbord: $INDEX_PATH"

    if [ $NO_BROWSER -eq 0 ]; then
        if command -v xdg-open &>/dev/null; then
            xdg-open "$INDEX_PATH" &>/dev/null &
        elif command -v open &>/dev/null; then
            open "$INDEX_PATH" &>/dev/null &
        fi
    fi
fi
