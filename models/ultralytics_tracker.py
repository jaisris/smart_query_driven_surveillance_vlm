"""Combined detection + tracking backend: YOLO.predict() + a lightweight IOU tracker.

Originally this wrapped YOLO's built-in track() mode (ByteTrack/BoT-SORT via
persist=True). That was replaced after debugging a real bounding-box-loss bug:
Ultralytics' persist=True tracking assumes near-continuous video (~33ms between
frames). On long videos we only run detection on every Nth raw frame (adaptive
frame skip commonly spaces samples 0.5-0.7s apart), and this breaks the
tracker's Kalman/gating logic badly enough that it silently drops most new
(non-persistent) object detections shortly after the very first call — verified
empirically: a fresh model finds a person at conf=0.87 in isolation, but the
SAME frame content produces zero person detections once the model has made
even one prior track() call, and the degradation carries over even to a
subsequent predict() call on that "warmed up" model/predictor object. Plain
predict() called repeatedly (no persist state) stayed 100% reliable across
285 consecutive calls in testing.

So detection here always uses predict(), and continuity across sampled frames
is provided by our own simple greedy IOU match against the immediately
preceding sampled frame (same class only) — no Kalman filter, since one
wouldn't meaningfully help at this frame spacing anyway. A static object
(e.g. a parked car) keeps a stable ID because its box barely moves between
samples; a person walking through keeps an ID for as many consecutive sampled
frames as they remain in roughly the same place, and gets a fresh ID once
they've moved on — an honest reflection of what sparse sampling can support,
rather than a persistent identity the sampling rate can't actually back up.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import Detection, Track, TrackSnapshot

logger = get_logger(__name__)

_VALID_BACKENDS = ("bytetrack", "botsort")


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class UltralyticsTracker:
    """YOLO.predict() detection + a simple frame-to-frame IOU tracker."""

    IOU_MATCH_THRESH = 0.3

    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        backend = self.config.tracking.backend
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"Unknown ultralytics tracker backend: {backend}")
        self._conf = self.config.yolo.confidence_threshold
        self._iou = self.config.yolo.iou_threshold
        self._classes = list(self.config.yolo.class_whitelist) or None

        device_str = self.config.yolo.device
        if device_str == "auto":
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device_str

        logger.info("Loading YOLO '%s' (predict + IOU-linked tracking) on %s ...",
                    self.config.yolo.model, self._device)
        self._model = YOLO(self.config.yolo.model)
        self._histories: Dict[int, List[TrackSnapshot]] = {}
        # (track_id, class_name, bbox_xyxy) from the previous processed frame
        self._active: List[Tuple[int, str, List[float]]] = []
        self._next_id = 1
        logger.info("Ultralytics tracker initialised (predict-only + IOU linking).")

    def detect_and_track(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Tuple[List[Detection], List[Track]]:
        """Run detection on one frame and link it to the previous frame's boxes."""
        result = self._model.predict(
            frame_bgr,
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
            self._active = []
            return detections, tracks

        candidates: List[Tuple[List[float], str]] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            cls_name = result.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(bbox_xyxy=xyxy, class_id=cls_id, class_name=cls_name, confidence=conf)
            )
            candidates.append((xyxy, cls_name))

        # Greedy highest-IOU-first matching against the previous frame's boxes,
        # restricted to the same class.
        pairs: List[Tuple[float, int, int]] = []
        for ci, (xyxy, cls_name) in enumerate(candidates):
            for pi, (_, pcls, pbbox) in enumerate(self._active):
                if pcls != cls_name:
                    continue
                score = _iou(xyxy, pbbox)
                if score >= self.IOU_MATCH_THRESH:
                    pairs.append((score, ci, pi))
        pairs.sort(key=lambda p: p[0], reverse=True)

        assigned_id: List[Optional[int]] = [None] * len(candidates)
        used_prev: set = set()
        for _, ci, pi in pairs:
            if assigned_id[ci] is not None or pi in used_prev:
                continue
            assigned_id[ci] = self._active[pi][0]
            used_prev.add(pi)

        new_active: List[Tuple[int, str, List[float]]] = []
        for ci, (xyxy, cls_name) in enumerate(candidates):
            track_id = assigned_id[ci]
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
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
            new_active.append((track_id, cls_name, xyxy))
        self._active = new_active

        return detections, tracks

    def get_track_histories(self) -> Dict[int, List[TrackSnapshot]]:
        return dict(self._histories)

    def reset(self) -> None:
        """Clear histories and linking state for a new video."""
        self._histories.clear()
        self._active = []
        self._next_id = 1
