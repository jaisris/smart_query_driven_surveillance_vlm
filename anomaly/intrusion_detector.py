"""Rule-based intrusion detection using ROI polygon zones.

Detects when a track's centroid enters any forbidden zone defined in config
as a list of pixel-coordinate polygons. Uses cv2.pointPolygonTest.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils.types import AnomalyEvent, TrackSnapshot


def detect_intrusion(
    track_histories: Dict[int, List[TrackSnapshot]],
    roi_polygons: List[List[List[int]]],
    fps: float,
) -> List[AnomalyEvent]:
    if not roi_polygons:
        return []

    # Convert polygon lists to numpy int32 arrays once
    np_polygons = [
        np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        for poly in roi_polygons
    ]

    events: List[AnomalyEvent] = []
    for track_id, snapshots in track_histories.items():
        events.extend(_scan_track(track_id, snapshots, np_polygons))
    return events


def _scan_track(
    track_id: int,
    snapshots: List[TrackSnapshot],
    np_polygons: List[np.ndarray],
) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    in_zone = False
    entry_snap: TrackSnapshot | None = None
    last_snap: TrackSnapshot | None = None

    for snap in snapshots:
        point = (float(snap.centroid_xy[0]), float(snap.centroid_xy[1]))
        inside = any(
            cv2.pointPolygonTest(poly, point, measureDist=False) >= 0
            for poly in np_polygons
        )

        if inside and not in_zone:
            in_zone = True
            entry_snap = snap
        elif not inside and in_zone:
            in_zone = False
            if entry_snap is not None:
                events.append(
                    AnomalyEvent(
                        track_id=track_id,
                        event_type="intrusion",
                        start_sec=entry_snap.timestamp_sec,
                        end_sec=last_snap.timestamp_sec if last_snap else entry_snap.timestamp_sec,
                        location_xy=entry_snap.centroid_xy,
                        severity=1.0,
                    )
                )
                entry_snap = None

        if inside:
            last_snap = snap

    # Close any open intrusion at end of video
    if in_zone and entry_snap is not None and last_snap is not None:
        events.append(
            AnomalyEvent(
                track_id=track_id,
                event_type="intrusion",
                start_sec=entry_snap.timestamp_sec,
                end_sec=last_snap.timestamp_sec,
                location_xy=entry_snap.centroid_xy,
                severity=1.0,
            )
        )
    return events
