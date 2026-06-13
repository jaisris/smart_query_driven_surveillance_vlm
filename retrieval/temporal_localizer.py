"""Merge top-K search results into contiguous temporal video segments."""

from __future__ import annotations

from typing import List

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import SearchResult, VideoSegment

logger = get_logger(__name__)


def localize_segments(
    results: List[SearchResult],
    gap_threshold_sec: float | None = None,
    min_segment_duration_sec: float | None = None,
    config: AppConfig | None = None,
) -> List[VideoSegment]:
    """Merge SearchResults whose timestamps are within gap_threshold_sec of each other.

    Returns VideoSegments sorted by peak cosine score (descending).
    """
    cfg = config or get_config()
    gap = gap_threshold_sec if gap_threshold_sec is not None else cfg.retrieval.gap_threshold_sec
    min_dur = (
        min_segment_duration_sec
        if min_segment_duration_sec is not None
        else cfg.retrieval.min_segment_duration_sec
    )

    if not results:
        return []

    # Sort by timestamp
    sorted_results = sorted(results, key=lambda r: r.timestamp_sec)

    segments: List[VideoSegment] = []
    current: List[SearchResult] = [sorted_results[0]]

    for r in sorted_results[1:]:
        if r.timestamp_sec - current[-1].timestamp_sec <= gap:
            current.append(r)
        else:
            segments.append(_make_segment(current))
            current = [r]
    segments.append(_make_segment(current))

    # Filter by minimum duration and sort by peak score
    segments = [s for s in segments if s.duration_sec >= min_dur or len(s.frame_indices) == 1]
    segments.sort(key=lambda s: s.peak_score, reverse=True)
    logger.info(
        "Localised %d segments from %d search results (gap=%.1fs, min_dur=%.1fs)",
        len(segments), len(results), gap, min_dur,
    )
    return segments


def _make_segment(results: List[SearchResult]) -> VideoSegment:
    start = results[0].timestamp_sec
    end = results[-1].timestamp_sec
    peak = max(r.cosine_score for r in results)
    indices = [r.frame_index for r in results]
    return VideoSegment(
        start_sec=start,
        end_sec=end,
        peak_score=peak,
        frame_indices=indices,
    )
