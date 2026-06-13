"""Tests for SimilaritySearch FAISS index build + query."""

import tempfile
import os

import numpy as np
import pytest

from retrieval.similarity_search import SimilaritySearch
from utils.config_loader import AppConfig
from utils.types import FrameIndexEntry


def _l2_norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _make_entries(n):
    return [FrameIndexEntry(frame_index=i, timestamp_sec=i / 25.0, track_ids=[]) for i in range(n)]


def test_build_and_search_top_k():
    n, dim = 50, 512
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    entries = _make_entries(n)

    search = SimilaritySearch(AppConfig())
    search.build_index(embeddings, entries)

    query = _l2_norm(np.random.randn(dim).astype(np.float32))
    results = search.search(query, top_k=5)

    assert len(results) == 5
    # Results should be sorted descending by score
    scores = [r.cosine_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_returns_correct_best_match():
    n, dim = 20, 512
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    entries = _make_entries(n)

    # Make frame 7 identical to the query — should be top result
    query = embeddings[7].copy()
    search = SimilaritySearch(AppConfig())
    search.build_index(embeddings, entries)
    results = search.search(query, top_k=3)

    assert results[0].frame_index == 7
    assert results[0].cosine_score == pytest.approx(1.0, abs=1e-4)


def test_save_and_load_index():
    n, dim = 10, 512
    embeddings = np.random.randn(n, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    entries = _make_entries(n)

    search = SimilaritySearch(AppConfig())
    search.build_index(embeddings, entries)

    with tempfile.TemporaryDirectory() as d:
        idx_path = os.path.join(d, "test.faiss")
        search.save_index(idx_path)

        search2 = SimilaritySearch(AppConfig())
        search2.load_index(idx_path, entries)
        query = _l2_norm(np.random.randn(dim).astype(np.float32))
        r1 = search.search(query, top_k=3)
        r2 = search2.search(query, top_k=3)
        assert [r.frame_index for r in r1] == [r.frame_index for r in r2]


def test_raises_without_index():
    search = SimilaritySearch(AppConfig())
    with pytest.raises(RuntimeError):
        search.search(np.zeros(512, dtype=np.float32), top_k=5)
