"""Aggregates all anomaly detectors and returns a unified event list."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from anomaly.intrusion_detector import detect_intrusion
from anomaly.loitering_detector import detect_loitering
from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import AnomalyEvent, FrameIndexEntry, TrackSnapshot

logger = get_logger(__name__)


class AnomalyEngine:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()

    def analyze(
        self,
        track_histories: Dict[int, List[TrackSnapshot]],
        fps: float,
        frame_embeddings: Optional[np.ndarray] = None,
        frame_index_entries: Optional[List[FrameIndexEntry]] = None,
    ) -> List[AnomalyEvent]:
        """Run all enabled anomaly detectors and return merged, sorted events."""
        events: List[AnomalyEvent] = []
        cfg = self.config.anomaly

        if cfg.enable_rule_based:
            # Loitering
            loitering_events = detect_loitering(
                track_histories=track_histories,
                fps=fps,
                dwell_radius_px=cfg.loitering.dwell_radius_px,
                dwell_time_sec=cfg.loitering.dwell_time_sec,
                person_only=cfg.loitering.person_only,
            )
            logger.info("Rule-based: %d loitering events", len(loitering_events))
            events.extend(loitering_events)

            # Intrusion
            intrusion_events = detect_intrusion(
                track_histories=track_histories,
                roi_polygons=cfg.intrusion.roi_zones,
                fps=fps,
            )
            logger.info("Rule-based: %d intrusion events", len(intrusion_events))
            events.extend(intrusion_events)

        if cfg.enable_vadclip and frame_embeddings is not None and frame_index_entries is not None:
            try:
                from anomaly.vadclip_detector import VadCLIPDetector, vadclip_to_anomaly_events
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                detector = VadCLIPDetector(weights_path=cfg.vadclip_weights, device=device)
                scores = detector.score_video(frame_embeddings)
                vc_events = vadclip_to_anomaly_events(scores, frame_index_entries)
                logger.info("VadCLIP: %d anomaly events", len(vc_events))
                events.extend(vc_events)
            except Exception as e:
                logger.warning("VadCLIP failed: %s", e)

        # Deduplicate: if two events of the same type and track_id overlap in time, keep the higher-severity one
        events = _deduplicate(events)
        events.sort(key=lambda e: e.start_sec)
        return events


def _deduplicate(events: List[AnomalyEvent]) -> List[AnomalyEvent]:
    """Remove overlapping events of the same (track_id, event_type), keeping highest severity."""
    groups: Dict[tuple, List[AnomalyEvent]] = {}
    for e in events:
        key = (e.track_id, e.event_type)
        groups.setdefault(key, []).append(e)

    result: List[AnomalyEvent] = []
    for group in groups.values():
        group.sort(key=lambda e: e.start_sec)
        merged = [group[0]]
        for e in group[1:]:
            prev = merged[-1]
            # Overlap if e starts before prev ends
            if e.start_sec <= prev.end_sec:
                # Extend and take higher severity
                merged[-1] = AnomalyEvent(
                    track_id=prev.track_id,
                    event_type=prev.event_type,
                    start_sec=min(prev.start_sec, e.start_sec),
                    end_sec=max(prev.end_sec, e.end_sec),
                    location_xy=prev.location_xy,
                    severity=max(prev.severity, e.severity),
                )
            else:
                merged.append(e)
        result.extend(merged)
    return result
