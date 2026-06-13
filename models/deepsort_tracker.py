"""DeepSORT multi-object tracking wrapper.

Maintains track histories (list of TrackSnapshot per track_id) which the
anomaly engine consumes after a full video is processed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import Detection, Track, TrackSnapshot

logger = get_logger(__name__)


class DeepSORTTracker:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        cfg = self.config.deepsort

        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            max_iou_distance=cfg.max_iou_distance,
            max_cosine_distance=cfg.max_cosine_distance,
            nn_budget=cfg.nn_budget,
        )
        # track_id → list of snapshots
        self._histories: Dict[int, List[TrackSnapshot]] = {}
        logger.info("DeepSORT tracker initialised.")

    def update(
        self,
        detections: List[Detection],
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
    ) -> List[Track]:
        """Feed detections into the tracker. Returns confirmed tracks."""
        # deep_sort_realtime expects detections as list of ([left,top,w,h], conf, class)
        raw_dets = []
        det_classes = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox_xyxy
            w, h = x2 - x1, y2 - y1
            raw_dets.append(([x1, y1, w, h], d.confidence, d.class_name))
            det_classes.append(d.class_name)

        tracks_raw = self._tracker.update_tracks(raw_dets, frame=frame_bgr)

        tracks: List[Track] = []
        for t in tracks_raw:
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()   # [left, top, right, bottom]
            bbox_xyxy = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
            cls_name = t.det_class if hasattr(t, "det_class") and t.det_class else "unknown"
            track = Track(
                track_id=t.track_id,
                bbox_xyxy=bbox_xyxy,
                class_name=cls_name,
                age_frames=t.age,
                is_confirmed=True,
            )
            tracks.append(track)

            # Record snapshot for anomaly engine
            cx, cy = track.centroid
            snapshot = TrackSnapshot(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                centroid_xy=(cx, cy),
                bbox_xyxy=bbox_xyxy,
                class_name=cls_name,
            )
            self._histories.setdefault(t.track_id, []).append(snapshot)

        return tracks

    def get_track_histories(self) -> Dict[int, List[TrackSnapshot]]:
        return dict(self._histories)

    def reset(self) -> None:
        """Clear histories for a new video."""
        self._histories.clear()
        from deep_sort_realtime.deepsort_tracker import DeepSort
        cfg = self.config.deepsort
        self._tracker = DeepSort(
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            max_iou_distance=cfg.max_iou_distance,
            max_cosine_distance=cfg.max_cosine_distance,
            nn_budget=cfg.nn_budget,
        )
