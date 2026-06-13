"""Tests for loitering, intrusion, and AnomalyEngine with synthetic track histories."""

import math

import numpy as np
import pytest

from anomaly.loitering_detector import detect_loitering
from anomaly.intrusion_detector import detect_intrusion
from anomaly.anomaly_engine import AnomalyEngine, _deduplicate
from utils.config_loader import AppConfig
from utils.types import AnomalyEvent, TrackSnapshot


def _snap(frame_idx, ts, cx, cy, cls="person"):
    return TrackSnapshot(
        frame_index=frame_idx,
        timestamp_sec=ts,
        centroid_xy=(cx, cy),
        bbox_xyxy=[cx - 20, cy - 40, cx + 20, cy + 40],
        class_name=cls,
    )


def _static_track(n=300, fps=25.0, cx=200.0, cy=300.0):
    """Track that stays in place for n/fps seconds."""
    return [_snap(i, i / fps, cx + np.random.uniform(-10, 10), cy + np.random.uniform(-10, 10)) for i in range(n)]


def _moving_track(n=100, fps=25.0):
    """Track that moves continuously — should NOT trigger loitering."""
    return [_snap(i, i / fps, float(i * 10), 300.0) for i in range(n)]


# ------------------------------------------------------------------ #
#  Loitering
# ------------------------------------------------------------------ #

def test_loitering_detected_when_dwell_exceeded():
    histories = {1: _static_track(n=300, fps=25.0)}  # 12 seconds
    events = detect_loitering(histories, fps=25.0, dwell_radius_px=80, dwell_time_sec=10.0)
    assert len(events) >= 1
    assert events[0].event_type == "loitering"
    assert events[0].track_id == 1
    assert events[0].severity >= 1.0


def test_loitering_not_triggered_for_moving_track():
    histories = {2: _moving_track(n=100, fps=25.0)}
    events = detect_loitering(histories, fps=25.0, dwell_radius_px=80, dwell_time_sec=10.0)
    assert len(events) == 0


def test_loitering_severity_proportional_to_dwell():
    histories = {3: _static_track(n=500, fps=25.0)}  # 20 seconds
    events = detect_loitering(histories, fps=25.0, dwell_radius_px=80, dwell_time_sec=10.0)
    assert events[0].severity == pytest.approx(2.0, abs=0.5)


# ------------------------------------------------------------------ #
#  Intrusion
# ------------------------------------------------------------------ #

def test_intrusion_detected_when_entering_zone():
    roi = [[[100, 100], [400, 100], [400, 400], [100, 400]]]
    # Track starts outside (x=50), enters zone at ~frame 12 (x=100)
    snaps = [_snap(i, i / 25.0, float(50 + i * 4), 250.0) for i in range(100)]
    events = detect_intrusion({1: snaps}, roi_polygons=roi, fps=25.0)
    assert len(events) >= 1
    assert events[0].event_type == "intrusion"


def test_intrusion_not_triggered_outside_zone():
    roi = [[[500, 500], [800, 500], [800, 800], [500, 800]]]
    snaps = [_snap(i, i / 25.0, float(i * 2), 250.0) for i in range(100)]
    events = detect_intrusion({1: snaps}, roi_polygons=roi, fps=25.0)
    assert len(events) == 0


def test_no_intrusion_with_empty_roi():
    snaps = [_snap(i, i / 25.0, 200.0, 200.0) for i in range(50)]
    events = detect_intrusion({1: snaps}, roi_polygons=[], fps=25.0)
    assert events == []


# ------------------------------------------------------------------ #
#  AnomalyEngine
# ------------------------------------------------------------------ #

def test_anomaly_engine_runs_without_crash():
    config = AppConfig()
    config.anomaly.enable_vadclip = False
    engine = AnomalyEngine(config)
    histories = {
        1: _static_track(n=300),
        2: _moving_track(n=100),
    }
    events = engine.analyze(histories, fps=25.0)
    # Just ensure it returns a list
    assert isinstance(events, list)


def test_deduplicate_merges_overlapping_events():
    e1 = AnomalyEvent(1, "loitering", 0.0, 10.0, (0.0, 0.0), severity=1.5)
    e2 = AnomalyEvent(1, "loitering", 5.0, 15.0, (0.0, 0.0), severity=2.0)
    result = _deduplicate([e1, e2])
    assert len(result) == 1
    assert result[0].end_sec == pytest.approx(15.0)
    assert result[0].severity == pytest.approx(2.0)


def test_deduplicate_keeps_non_overlapping():
    e1 = AnomalyEvent(1, "loitering", 0.0, 5.0, (0.0, 0.0), severity=1.0)
    e2 = AnomalyEvent(1, "loitering", 10.0, 15.0, (0.0, 0.0), severity=1.0)
    result = _deduplicate([e1, e2])
    assert len(result) == 2
