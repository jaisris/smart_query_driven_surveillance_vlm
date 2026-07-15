"""Combined detection + tracking backend using Ultralytics built-in trackers.

Wraps YOLO's `track()` mode (ByteTrack or BoT-SORT), which runs detection and
association in a single call. Produces the same Track / TrackSnapshot histories
interface as DeepSORTTracker, so FrameProcessor and the anomaly engine are
agnostic to which backend is active (config: tracking.backend).

ByteTrack reaches 80.3 MOTA on MOT17 vs DeepSORT's 75.4, with no separate
re-ID network and no extra dependency.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import Detection, Track, TrackSnapshot

logger = get_logger(__name__)

_TRACKER_YAML = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


class UltralyticsTracker:
    """Detection + tracking in one call via YOLO.track(persist=True)."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        backend = self.config.tracking.backend
        if backend not in _TRACKER_YAML:
            raise ValueError(f"Unknown ultralytics tracker backend: {backend}")
        self._tracker_yaml = _TRACKER_YAML[backend]
        self._conf = self.config.yolo.confidence_threshold
        self._iou = self.config.yolo.iou_threshold
        self._classes = list(self.config.yolo.class_whitelist) or None

        device_str = self.config.yolo.device
        if device_str == "auto":
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device_str

        logger.info("Loading YOLO '%s' with %s tracking on %s ...",
                    self.config.yolo.model, backend, self._device)
        self._model = YOLO(self.config.yolo.model)
        self._histories: Dict[int, List[TrackSnapshot]] = {}
        logger.info("Ultralytics tracker initialised (%s).", self._tracker_yaml)

    def detect_and_track(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Tuple[List[Detection], List[Track]]:
        """Run detection + tracking on one frame. Returns (detections, tracks)."""
        result = self._model.track(
            frame_bgr,
            persist=True,
            tracker=self._tracker_yaml,
            conf=self._conf,
            iou=self._iou,
            classes=self._classes,
            device=self._device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        tracks: List[Track] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections, tracks

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            cls_name = result.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(bbox_xyxy=xyxy, class_id=cls_id, class_name=cls_name, confidence=conf)
            )
            if boxes.id is None:
                continue  # tracker not yet confirmed on early frames
            track_id = int(boxes.id[i])
            track = Track(
                track_id=track_id,
                bbox_xyxy=xyxy,
                class_name=cls_name,
                age_frames=len(self._histories.get(track_id, [])) + 1,
                is_confirmed=True,
            )
            tracks.append(track)
            cx, cy = track.centroid
            self._histories.setdefault(track_id, []).append(
                TrackSnapshot(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    centroid_xy=(cx, cy),
                    bbox_xyxy=xyxy,
                    class_name=cls_name,
                )
            )
        return detections, tracks

    def get_track_histories(self) -> Dict[int, List[TrackSnapshot]]:
        return dict(self._histories)

    def reset(self) -> None:
        """Clear histories and tracker state for a new video."""
        self._histories.clear()
        # Reset the internal tracker state; fall back to model reload if the
        # predictor has not been created yet or the API changes.
        try:
            if self._model.predictor is not None and getattr(self._model.predictor, "trackers", None):
                for t in self._model.predictor.trackers:
                    t.reset()
        except Exception:
            self._model = YOLO(self.config.yolo.model)
