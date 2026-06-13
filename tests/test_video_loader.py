"""Tests for VideoLoader."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from data.video_loader import VideoLoader
from utils.config_loader import AppConfig, PipelineConfig, VideoConfig


def _make_synthetic_video(path: str, n_frames: int = 30, fps: int = 30):
    """Write a tiny synthetic video to path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (64, 64))
    for i in range(n_frames):
        frame = np.full((64, 64, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def sample_video():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    _make_synthetic_video(path, n_frames=30, fps=30)
    yield path
    os.unlink(path)


def _config(frame_skip=1):
    cfg = AppConfig()
    cfg.pipeline = PipelineConfig(frame_skip=frame_skip)
    cfg.video = VideoConfig(max_resolution=[1280, 720])
    return cfg


def test_metadata(sample_video):
    loader = VideoLoader(sample_video, _config())
    meta = loader.get_metadata()
    assert meta.total_frames == 30
    assert meta.width == 64
    assert meta.height == 64
    assert meta.fps == pytest.approx(30, abs=1)


def test_iter_frames_all(sample_video):
    loader = VideoLoader(sample_video, _config(frame_skip=1))
    frames = list(loader.iter_frames())
    assert len(frames) == 30
    fi, ts, frame = frames[0]
    assert fi == 0
    assert ts == pytest.approx(0.0, abs=0.01)
    assert frame.shape == (64, 64, 3)


def test_iter_frames_skip(sample_video):
    loader = VideoLoader(sample_video, _config(frame_skip=5))
    frames = list(loader.iter_frames())
    assert len(frames) == 6   # 0, 5, 10, 15, 20, 25

    indices = [fi for fi, _, _ in frames]
    assert indices == [0, 5, 10, 15, 20, 25]


def test_timestamps_match_frame_index(sample_video):
    loader = VideoLoader(sample_video, _config(frame_skip=1))
    meta = loader.get_metadata()
    for fi, ts, _ in loader.iter_frames():
        expected_ts = fi / meta.fps
        assert ts == pytest.approx(expected_ts, abs=0.01)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        VideoLoader("/nonexistent/video.mp4")
