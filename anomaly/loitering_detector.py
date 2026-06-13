"""Rule-based loitering detection from DeepSORT track histories.

A track is flagged as loitering when its centroid stays within a circle of
radius `dwell_radius_px` for longer than `dwell_time_sec` seconds.
Severity = actual_dwell_sec / dwell_time_sec.
"""

from __future__ import annotations

import math
from typing import Dict, List

from utils.logger import get_logger
from utils.types import AnomalyEvent, TrackSnapshot

logger = get_logger(__name__)


def detect_loitering(
    track_histories: Dict[int, List[TrackSnapshot]],
    fps: float,
    dwell_radius_px: int = 80,
    dwell_time_sec: float = 30.0,
    person_only: bool = True,
) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    skipped = 0
    for track_id, snapshots in track_histories.items():
        if len(snapshots) < 2:
            continue
        if person_only and snapshots and snapshots[0].class_name.lower() != "person":
            skipped += 1
            continue
        track_events = _scan_track(track_id, snapshots, fps, dwell_radius_px, dwell_time_sec)
        if track_events:
            logger.debug(
                f"Track {track_id}: {len(track_events)} loitering event(s) "
                f"(radius={dwell_radius_px}px, threshold={dwell_time_sec:.1f}s)"
            )
        events.extend(track_events)
    logger.info(
        f"Loitering scan: {len(track_histories)} tracks ({skipped} non-person skipped), "
        f"{len(events)} events (radius={dwell_radius_px}px, threshold={dwell_time_sec:.1f}s)"
    )
    return events


def _scan_track(
    track_id: int,
    snapshots: List[TrackSnapshot],
    fps: float,
    radius_px: int,
    min_dwell_sec: float,
) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    anchor_idx = 0

    while anchor_idx < len(snapshots):
        anchor = snapshots[anchor_idx]
        ax, ay = anchor.centroid_xy
        end_idx = anchor_idx

        for j in range(anchor_idx + 1, len(snapshots)):
            cx, cy = snapshots[j].centroid_xy
            dist = math.sqrt((cx - ax) ** 2 + (cy - ay) ** 2)
            if dist <= radius_px:
                end_idx = j
            else:
                break

        if end_idx > anchor_idx:
            dwell_sec = snapshots[end_idx].timestamp_sec - anchor.timestamp_sec
            if dwell_sec >= min_dwell_sec:
                severity = dwell_sec / min_dwell_sec
                events.append(
                    AnomalyEvent(
                        track_id=track_id,
                        event_type="loitering",
                        start_sec=anchor.timestamp_sec,
                        end_sec=snapshots[end_idx].timestamp_sec,
                        location_xy=(ax, ay),
                        severity=round(severity, 2),
                    )
                )
        anchor_idx = end_idx + 1

    return events
