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


def _static_signature(frame_rgb: np.ndarray) -> np.ndarray:
    """Tiny grayscale thumbnail used to compare consecutive frames for static content."""
    import cv2
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (64, 64)).astype(np.float32)


class EmbeddingBuilder:
    def __init__(self, encoder: CLIPEncoder, cache: CacheManager):
        self.encoder = encoder
        self.cache = cache
        self._stream_state: dict | None = None

    # ------------------------------------------------------------------
    # Streaming API — constant memory, used for long videos.
    # The pipeline calls stream_add() once per frame and stream_finalize()
    # at the end; pixel data is dropped as soon as each batch is encoded.
    # ------------------------------------------------------------------

    def stream_start(
        self,
        batch_size: int = 32,
        skip_static: bool = True,
        static_diff_threshold: float = 2.0,
    ) -> None:
        """Begin a streaming encode session."""
        self._stream_state = {
            "batch_size": batch_size,
            "skip_static": skip_static,
            "static_threshold": static_diff_threshold,
            "buffer_rgb": [],          # pending frames for the next CLIP batch
            "buffer_meta": [],         # (frame_index, timestamp_sec, track_ids)
            "embeddings": [],          # encoded (512,) vectors
            "index_entries": [],
            "last_signature": None,    # thumbnail of the last *encoded* frame
            "skipped_degenerate": 0,
            "skipped_static": 0,
        }

    def stream_add(self, processed_frame: ProcessedFrame) -> None:
        """Feed one frame into the streaming session. Encodes a batch when full."""
        s = self._stream_state
        if s is None:
            raise RuntimeError("stream_start() must be called before stream_add()")

        frame_rgb = processed_frame.frame_rgb
        if _is_degenerate_frame(frame_rgb):
            s["skipped_degenerate"] += 1
            return

        if s["skip_static"]:
            sig = _static_signature(frame_rgb)
            last = s["last_signature"]
            if last is not None:
                # Fraction (%) of thumbnail pixels that changed noticeably. This detects
                # small moving objects (a person is ~0.4% of a wide surveillance shot)
                # that a mean-difference test would average away, while staying immune
                # to codec noise, which rarely shifts a downscaled pixel by >12 levels.
                changed_pct = float((np.abs(sig - last) > 12.0).mean()) * 100.0
                if changed_pct < s["static_threshold"]:
                    s["skipped_static"] += 1
                    return
            s["last_signature"] = sig

        s["buffer_rgb"].append(frame_rgb)
        s["buffer_meta"].append(
            (
                processed_frame.frame_index,
                processed_frame.timestamp_sec,
                [t.track_id for t in processed_frame.tracks],
            )
        )
        if len(s["buffer_rgb"]) >= s["batch_size"]:
            self._stream_flush()

    def stream_finalize(self, cache_key: str) -> tuple[np.ndarray, List[FrameIndexEntry]]:
        """Encode any remaining frames, save to cache, and return the results."""
        s = self._stream_state
        if s is None:
            raise RuntimeError("stream_start() must be called before stream_finalize()")

        self._stream_flush()

        if s["skipped_degenerate"] or s["skipped_static"]:
            logger.info(
                "Streaming encode skipped %d degenerate + %d static frame(s)",
                s["skipped_degenerate"], s["skipped_static"],
            )

        if not s["embeddings"]:
            logger.warning("Streaming encode produced no embeddings — video may be blank/static")
            embedding_matrix = np.zeros((0, self.encoder.embedding_dim), dtype=np.float32)
            index_entries: List[FrameIndexEntry] = []
        else:
            embedding_matrix = np.stack(s["embeddings"], axis=0)  # (N, 512)
            index_entries = s["index_entries"]
            self.cache.save(cache_key, embedding_matrix, index_entries)

        self._stream_state = None
        return embedding_matrix, index_entries

    def _stream_flush(self) -> None:
        """Encode the pending buffer and drop its pixel data."""
        s = self._stream_state
        if not s["buffer_rgb"]:
            return
        vecs = self.encoder.encode_image_batch(s["buffer_rgb"])
        for vec, (frame_index, timestamp_sec, track_ids) in zip(vecs, s["buffer_meta"]):
            s["embeddings"].append(vec)
            s["index_entries"].append(
                FrameIndexEntry(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    track_ids=track_ids,
                )
            )
        s["buffer_rgb"].clear()
        s["buffer_meta"].clear()

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
