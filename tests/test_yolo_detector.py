"""Tests for YOLODetector — mocks ultralytics to avoid loading weights."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.types import Detection


def _make_mock_detector(mock_boxes=None):
    """Build a YOLODetector with a mocked ultralytics.YOLO."""
    with patch("models.yolo_detector.YOLO") as MockYOLO:
        mock_model = MagicMock()

        if mock_boxes:
            import torch
            mock_result = MagicMock()
            mock_result.names = {0: "person", 2: "car"}
            mock_result.boxes = MagicMock()
            mock_result.boxes.__iter__ = MagicMock(return_value=iter(mock_boxes))
            mock_model.predict.return_value = [mock_result]
        else:
            mock_result = MagicMock()
            mock_result.boxes = None
            mock_model.predict.return_value = [mock_result]

        MockYOLO.return_value = mock_model

        from models.yolo_detector import YOLODetector
        from utils.config_loader import AppConfig
        return YOLODetector(config=AppConfig())


def _mock_box(xyxy, conf, cls):
    import torch
    box = MagicMock()
    box.xyxy = [torch.tensor(xyxy)]
    box.conf = [torch.tensor(conf)]
    box.cls = [torch.tensor(cls)]
    return box


def test_detect_returns_list():
    det = _make_mock_detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(frame)
    assert isinstance(results, list)


def test_detect_parses_box_correctly():
    box = _mock_box([100.0, 150.0, 300.0, 400.0], conf=0.85, cls=0)
    det = _make_mock_detector(mock_boxes=[box])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(frame)
    assert len(results) == 1
    d: Detection = results[0]
    assert d.bbox_xyxy == pytest.approx([100.0, 150.0, 300.0, 400.0], abs=0.01)
    assert d.class_id == 0
    assert d.class_name == "person"
    assert d.confidence == pytest.approx(0.85, abs=0.01)


def test_centroid_computed_correctly():
    box = _mock_box([100.0, 100.0, 200.0, 300.0], conf=0.9, cls=0)
    det = _make_mock_detector(mock_boxes=[box])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = det.detect(frame)
    cx, cy = results[0].centroid
    assert cx == pytest.approx(150.0)
    assert cy == pytest.approx(200.0)
