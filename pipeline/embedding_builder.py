"""Builds CLIP embeddings for processed frames, with cache-first logic."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from data.cache_manager import CacheManager
from models.clip_encoder import CLIPEncoder
from utils.logger import get_logger
from utils.types import FrameEmbedding, FrameIndexEntry, ProcessedFrame

logger = get_logger(__name__)


def _is_degenerate_frame(frame_rgb: np.ndarray, min_mean: float = 8.0, min_std: float = 6.0) -> bool:
    """True for frames that carry no useful content: near-black, near-white, or flat.

    Such frames (e.g. a dropped/transition frame that decodes to solid black) still
    receive a valid CLIP embedding and can otherwise surface as spurious retrieval
    matches, so they are excluded from the index.
    """
    mean = float(frame_rgb.mean())
    std = float(frame_rgb.std())
    return std < min_std or mean < min_mean or mean > (255.0 - min_mean)


class EmbeddingBuilder:
    def __init__(self, encoder: CLIPEncoder, cache: CacheManager):
        self.encoder = encoder
        self.cache = cache

    def build(self, processed_frame: ProcessedFrame) -> FrameEmbedding:
        """Encode a single frame. RGB conversion has already happened in FrameProcessor."""
        vec = self.encoder.encode_image(processed_frame.frame_rgb)
        return FrameEmbedding(
            frame_index=processed_frame.frame_index,
            timestamp_sec=processed_frame.timestamp_sec,
            vector=vec,
        )

    def build_batch(self, frames: List[ProcessedFrame]) -> List[FrameEmbedding]:
        """Batch-encode frames (more GPU-efficient than one-at-a-time)."""
        if not frames:
            return []
        rgb_list = [f.frame_rgb for f in frames]
        vecs = self.encoder.encode_image_batch(rgb_list)
        return [
            FrameEmbedding(
                frame_index=f.frame_index,
                timestamp_sec=f.timestamp_sec,
                vector=vecs[i],
            )
            for i, f in enumerate(frames)
        ]

    def build_all_from_cache_or_encode(
        self,
        all_frames: List[ProcessedFrame],
        cache_key: str,
        batch_size: int = 32,
    ) -> tuple[np.ndarray, List[FrameIndexEntry]]:
        """Try loading from cache. If miss, encode in batches and save."""
        if self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        # Drop degenerate (black/blank) frames so they never enter the retrieval index.
        usable_frames = [f for f in all_frames if not _is_degenerate_frame(f.frame_rgb)]
        skipped = len(all_frames) - len(usable_frames)
        if skipped:
            logger.info("Skipped %d degenerate (black/blank) frame(s) before embedding", skipped)
        if not usable_frames:                      # safety: never index an empty video
            usable_frames = all_frames

        logger.info("Cache miss — encoding %d frames in batches of %d ...", len(usable_frames), batch_size)
        embeddings: List[np.ndarray] = []
        index_entries: List[FrameIndexEntry] = []

        for start in range(0, len(usable_frames), batch_size):
            batch = usable_frames[start:start + batch_size]
            batch_embs = self.build_batch(batch)
            for emb, pf in zip(batch_embs, batch):
                embeddings.append(emb.vector)
                index_entries.append(
                    FrameIndexEntry(
                        frame_index=pf.frame_index,
                        timestamp_sec=pf.timestamp_sec,
                        track_ids=[t.track_id for t in pf.tracks],
                    )
                )
            logger.info("  Encoded frames %d–%d", start, min(start + batch_size, len(usable_frames)))

        embedding_matrix = np.stack(embeddings, axis=0)   # (N, 512)
        self.cache.save(cache_key, embedding_matrix, index_entries)
        return embedding_matrix, index_entries
