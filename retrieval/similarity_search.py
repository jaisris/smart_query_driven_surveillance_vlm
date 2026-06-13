"""FAISS-based cosine similarity search over frame embeddings.

Uses IndexFlatIP (inner product) on L2-normalised vectors which equals cosine
similarity. Exact search is fine at this scale (~thousands of frames).
"""

from __future__ import annotations

import os
from typing import List

import faiss
import numpy as np

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import FrameIndexEntry, SearchResult

logger = get_logger(__name__)


class SimilaritySearch:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        self._index: faiss.IndexFlatIP | None = None
        self._frame_entries: List[FrameIndexEntry] = []

    def build_index(
        self,
        embedding_matrix: np.ndarray,
        frame_index_entries: List[FrameIndexEntry],
    ) -> None:
        """Build a FAISS flat index from an (N, 512) embedding matrix."""
        if embedding_matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {embedding_matrix.shape}")
        dim = embedding_matrix.shape[1]
        mat = embedding_matrix.astype(np.float32)
        # Ensure L2-normalised (should already be, but re-normalise for safety)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / (norms + 1e-8)

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(mat)
        self._frame_entries = frame_index_entries
        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)

    def search(self, query_vector: np.ndarray, top_k: int | None = None) -> List[SearchResult]:
        """Return top-K frames by cosine similarity to the query vector."""
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("Index not built. Call build_index() first.")
        k = top_k or self.config.retrieval.top_k
        k = min(k, self._index.ntotal)

        q = query_vector.astype(np.float32).reshape(1, -1)
        q /= np.linalg.norm(q) + 1e-8

        scores, indices = self._index.search(q, k)
        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._frame_entries):
                continue
            entry = self._frame_entries[idx]
            results.append(
                SearchResult(
                    frame_index=entry.frame_index,
                    timestamp_sec=entry.timestamp_sec,
                    cosine_score=float(score),
                )
            )
        if results:
            logger.info(
                "FAISS search: top_k=%d returned %d results — "
                "top score=%.3f (t=%.1fs)  lowest=%.3f (t=%.1fs)",
                k, len(results),
                results[0].cosine_score, results[0].timestamp_sec,
                results[-1].cosine_score, results[-1].timestamp_sec,
            )
        return results

    def save_index(self, path: str) -> None:
        if self._index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self._index, path)
        logger.info("FAISS index saved to %s", path)

    def load_index(self, path: str, frame_index_entries: List[FrameIndexEntry]) -> None:
        self._index = faiss.read_index(path)
        self._frame_entries = frame_index_entries
        logger.info("FAISS index loaded from %s (%d vectors)", path, self._index.ntotal)
