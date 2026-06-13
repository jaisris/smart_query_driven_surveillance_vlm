"""Saves and loads CLIP frame embeddings to/from disk.

Cache key = SHA256(video_path + frame_skip + clip_model_name) so the cache
auto-invalidates whenever any of those change.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from utils.types import FrameIndexEntry

logger = get_logger(__name__)


class CacheMissError(Exception):
    pass


class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def key(self, video_path: str, frame_skip: int, clip_model: str) -> str:
        raw = f"{os.path.abspath(video_path)}|{frame_skip}|{clip_model}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def content_key(self, file_content_hash: str, frame_skip: int, clip_model: str) -> str:
        """Cache key based on file content SHA256 — stable across re-uploads of the same video."""
        raw = f"content:{file_content_hash}|{frame_skip}|{clip_model}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def save(
        self,
        cache_key: str,
        embeddings: np.ndarray,
        index_entries: List[FrameIndexEntry],
    ) -> None:
        npy_path = self._npy_path(cache_key)
        idx_path = self._idx_path(cache_key)
        np.save(npy_path, embeddings)
        records = [
            {
                "frame_index": e.frame_index,
                "timestamp_sec": e.timestamp_sec,
                "track_ids": e.track_ids,
            }
            for e in index_entries
        ]
        with open(idx_path, "w") as f:
            json.dump(records, f)
        logger.info("Cache saved: %s (%d frames, shape %s)", cache_key, len(records), embeddings.shape)

    def load(
        self, cache_key: str
    ) -> Tuple[np.ndarray, List[FrameIndexEntry]]:
        npy_path = self._npy_path(cache_key)
        idx_path = self._idx_path(cache_key)
        if not os.path.exists(npy_path) or not os.path.exists(idx_path):
            raise CacheMissError(f"No cache found for key: {cache_key}")
        embeddings = np.load(npy_path)
        with open(idx_path, "r") as f:
            records = json.load(f)
        entries = [
            FrameIndexEntry(
                frame_index=r["frame_index"],
                timestamp_sec=r["timestamp_sec"],
                track_ids=r.get("track_ids", []),
            )
            for r in records
        ]
        logger.info("Cache hit: %s (%d frames)", cache_key, len(entries))
        return embeddings, entries

    def exists(self, cache_key: str) -> bool:
        return os.path.exists(self._npy_path(cache_key)) and os.path.exists(
            self._idx_path(cache_key)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def save_pipeline_state(self, cache_key: str, state: Dict[str, Any]) -> None:
        """Persist track_histories + VideoMetadata so Step 1 can be skipped on re-run."""
        pkl_path = self._tracks_path(cache_key)
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Track cache saved: %s", cache_key)

    def load_pipeline_state(self, cache_key: str) -> Optional[Dict[str, Any]]:
        pkl_path = self._tracks_path(cache_key)
        if not os.path.exists(pkl_path):
            return None
        try:
            with open(pkl_path, "rb") as f:
                state = pickle.load(f)
            logger.info("Track cache hit: %s", cache_key)
            return state
        except Exception as exc:
            logger.warning("Track cache load failed (%s): %s — will re-run pipeline", cache_key, exc)
            return None

    def _npy_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _idx_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}_index.json")

    def _tracks_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}_tracks.pkl")
