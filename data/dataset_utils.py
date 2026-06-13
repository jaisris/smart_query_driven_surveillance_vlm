"""Helpers for UCF-Crime / UCA dataset loading and video metadata."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2

from utils.logger import get_logger
from utils.types import VideoMetadata

logger = get_logger(__name__)

# UCF-Crime 13 anomaly class names
UCF_CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]


def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return VideoMetadata(
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
        path=video_path,
    )


def load_ucf_crime_labels(annotation_file: str) -> Dict[str, str]:
    """Load video-level labels from UCF-Crime annotation text file.

    Returns dict: {video_filename: label_class}
    Expected format per line: "Abuse/Abuse001_x264.mp4  1"
    """
    labels: Dict[str, str] = {}
    if not os.path.exists(annotation_file):
        logger.warning("UCF-Crime annotation file not found: %s", annotation_file)
        return labels
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_path = parts[0]
                label_str = parts[1]
                class_name = video_path.split("/")[0] if "/" in video_path else "Normal"
                labels[os.path.basename(video_path)] = class_name
    return labels


def load_uca_annotations(uca_json_path: str) -> List[Dict]:
    """Load UCA (UCF-Crime Annotation) temporal NL grounding annotations.

    Returns list of dicts with keys:
      video_id, sentence, start_sec, end_sec
    """
    import json
    if not os.path.exists(uca_json_path):
        logger.warning("UCA annotation file not found: %s", uca_json_path)
        return []
    with open(uca_json_path, "r") as f:
        data = json.load(f)
    annotations = []
    for entry in data:
        annotations.append({
            "video_id": entry.get("video_id", ""),
            "sentence": entry.get("sentence", ""),
            "start_sec": entry.get("start_time", 0.0),
            "end_sec": entry.get("end_time", 0.0),
        })
    return annotations


def list_videos(root_dir: str, extensions: Tuple[str, ...] = (".mp4", ".avi", ".mkv")) -> List[str]:
    """Recursively list all video files under root_dir."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)
