"""Tests for CacheManager: save/load round-trip and cache hit/miss logic."""

import os
import tempfile

import numpy as np
import pytest

from data.cache_manager import CacheManager, CacheMissError
from utils.types import FrameIndexEntry


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_entries(n):
    return [FrameIndexEntry(frame_index=i, timestamp_sec=i / 25.0, track_ids=[i % 3]) for i in range(n)]


def test_save_and_load(cache_dir):
    cm = CacheManager(cache_dir)
    key = "testkey123"
    embeddings = np.random.randn(10, 512).astype(np.float32)
    entries = _make_entries(10)

    cm.save(key, embeddings, entries)

    loaded_emb, loaded_entries = cm.load(key)
    assert loaded_emb.shape == (10, 512)
    np.testing.assert_allclose(loaded_emb, embeddings)
    assert len(loaded_entries) == 10
    assert loaded_entries[5].frame_index == 5
    assert loaded_entries[5].timestamp_sec == pytest.approx(5 / 25.0, abs=1e-4)


def test_exists(cache_dir):
    cm = CacheManager(cache_dir)
    key = "existskey"
    assert not cm.exists(key)
    cm.save(key, np.zeros((5, 512), dtype=np.float32), _make_entries(5))
    assert cm.exists(key)


def test_cache_miss_raises(cache_dir):
    cm = CacheManager(cache_dir)
    with pytest.raises(CacheMissError):
        cm.load("doesnotexist")


def test_cache_key_deterministic(cache_dir):
    cm = CacheManager(cache_dir)
    k1 = cm.key("video.mp4", 5, "openai/clip-vit-base-patch32")
    k2 = cm.key("video.mp4", 5, "openai/clip-vit-base-patch32")
    k3 = cm.key("video.mp4", 10, "openai/clip-vit-base-patch32")
    assert k1 == k2
    assert k1 != k3
