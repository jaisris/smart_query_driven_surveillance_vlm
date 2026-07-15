"""Per-frame processing: detection + tracking → ProcessedFrame.

Two tracker styles are supported:
  - Combined backends (UltralyticsTracker): one detect_and_track() call
  - DeepSORT baseline: separate YOLO detect() then tracker.update()
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from models.yolo_detector import YOLODetector
from utils.logger import get_logger
from utils.types import ProcessedFrame

logger = get_logger(__name__)


class FrameProcessor:
    def __init__(self, detector: Optional[YOLODetector], tracker):
        if detector is None and not hasattr(tracker, "detect_and_track"):
            raise ValueError("A detector is required unless the tracker is a combined backend")
        self.detector = detector
        self.tracker = tracker

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> ProcessedFrame:
        if hasattr(self.tracker, "detect_and_track"):
            detections, tracks = self.tracker.detect_and_track(
                frame_bgr, frame_index=frame_index, timestamp_sec=timestamp_sec
            )
        else:
            detections = self.detector.detect(frame_bgr)
            tracks = self.tracker.update(
                detections, frame_bgr, frame_index=frame_index, timestamp_sec=timestamp_sec
            )
        logger.debug(
            "frame %d (t=%.2fs): %d detections, %d active tracks",
            frame_index, timestamp_sec, len(detections), len(tracks),
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return ProcessedFrame(
            frame_bgr=frame_bgr,
            frame_rgb=frame_rgb,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            detections=detections,
            tracks=tracks,
        )
