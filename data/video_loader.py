"""OpenCV-based frame iterator for a video file."""

from __future__ import annotations

import os
from typing import Iterator, Tuple

import cv2
import numpy as np

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import VideoMetadata

logger = get_logger(__name__)


class VideoLoader:
    def __init__(self, video_path: str, config: AppConfig | None = None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.config = config or get_config()
        self._cap = None
        self._metadata: VideoMetadata | None = None

    def get_metadata(self) -> VideoMetadata:
        if self._metadata is not None:
            return self._metadata
        cap = cv2.VideoCapture(self.video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps if fps > 0 else 0.0
            self._metadata = VideoMetadata(
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                width=width,
                height=height,
                path=self.video_path,
            )
        finally:
            cap.release()
        return self._metadata

    def iter_frames(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield (frame_index, timestamp_sec, frame_bgr) for every Nth frame."""
        meta = self.get_metadata()
        skip = self.config.pipeline.frame_skip
        max_w, max_h = self.config.video.max_resolution

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frame_index = 0
        yielded = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_index % skip == 0:
                    h, w = frame.shape[:2]
                    if w > max_w or h > max_h:
                        scale = min(max_w / w, max_h / h)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    timestamp_sec = frame_index / meta.fps
                    yield frame_index, timestamp_sec, frame
                    yielded += 1
                frame_index += 1
        finally:
            cap.release()

        logger.info(
            "VideoLoader: %s — yielded %d frames (skip=%d, total=%d)",
            os.path.basename(self.video_path),
            yielded,
            skip,
            frame_index,
        )
