"""Per-frame processing: YOLO detection → DeepSORT tracking → ProcessedFrame."""

from __future__ import annotations

import cv2
import numpy as np

from models.deepsort_tracker import DeepSORTTracker
from models.yolo_detector import YOLODetector
from utils.logger import get_logger
from utils.types import ProcessedFrame

logger = get_logger(__name__)


class FrameProcessor:
    def __init__(self, detector: YOLODetector, tracker: DeepSORTTracker):
        self.detector = detector
        self.tracker = tracker

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> ProcessedFrame:
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
