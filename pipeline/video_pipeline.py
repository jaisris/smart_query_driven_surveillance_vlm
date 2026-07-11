"""Top-level pipeline orchestrator.

Single entry point for the Streamlit UI and notebooks:
    result = VideoPipeline().run("path/to/video.mp4")
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Optional

from tqdm import tqdm

from anomaly.anomaly_engine import AnomalyEngine
from data.cache_manager import CacheManager
from data.video_loader import VideoLoader
from models.clip_encoder import CLIPEncoder
from models.deepsort_tracker import DeepSORTTracker
from models.yolo_detector import YOLODetector
from pipeline.embedding_builder import EmbeddingBuilder
from pipeline.frame_processor import FrameProcessor
from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger, setup_logging
from utils.types import PipelineResult

setup_logging()
logger = get_logger(__name__)


class VideoPipeline:
    def __init__(
        self,
        config: AppConfig | None = None,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[DeepSORTTracker] = None,
        encoder: Optional[CLIPEncoder] = None,
    ):
        self.config = config or get_config()
        self._detector = detector
        self._tracker = tracker
        self._encoder = encoder

    def _get_detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(self.config)
        return self._detector

    def _get_tracker(self) -> DeepSORTTracker:
        if self._tracker is None:
            self._tracker = DeepSORTTracker(self.config)
        return self._tracker

    def _get_encoder(self) -> CLIPEncoder:
        if self._encoder is None:
            self._encoder = CLIPEncoder(self.config)
        return self._encoder

    def _effective_frame_skip(self, total_frames: int) -> int:
        """Adaptive skip: long videos raise frame_skip so at most
        pipeline.max_indexed_frames frames are processed."""
        base_skip = self.config.pipeline.frame_skip
        max_frames = self.config.pipeline.max_indexed_frames
        if max_frames <= 0:
            return base_skip
        adaptive = math.ceil(total_frames / max_frames)
        return max(base_skip, adaptive)

    def run(
        self,
        video_path: str,
        content_hash: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> PipelineResult:
        """Process a full video and return PipelineResult.

        Pass content_hash (SHA256 of file bytes) to enable content-based caching so that
        re-uploading the same video skips Steps 1 and 2 entirely.
        progress_callback(fraction, message) is invoked periodically for UI progress bars.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        t_start = time.time()
        logger.info("=== VideoPipeline.run START: '%s' ===", os.path.basename(video_path))

        encoder = self._get_encoder()
        cache = CacheManager(self.config.pipeline.cache_dir)
        embedding_builder = EmbeddingBuilder(encoder, cache)

        # Metadata is needed up front to compute the adaptive frame skip
        # (reading it only opens the container header — cheap even for 1-hour files).
        loader = VideoLoader(video_path, self.config)
        metadata = loader.get_metadata()
        effective_skip = self._effective_frame_skip(metadata.total_frames)
        if effective_skip != self.config.pipeline.frame_skip:
            logger.info(
                "Adaptive frame skip: %d → %d (video has %d frames, max_indexed_frames=%d)",
                self.config.pipeline.frame_skip, effective_skip,
                metadata.total_frames, self.config.pipeline.max_indexed_frames,
            )

        # --- Cache key: prefer content hash so re-uploads of the same file hit the cache ---
        if content_hash:
            cache_key = cache.content_key(
                content_hash,
                effective_skip,
                self.config.clip.model_name,
            )
        else:
            cache_key = cache.key(
                video_path,
                effective_skip,
                self.config.clip.model_name,
            )

        # --- Full cache check: if both track state AND embeddings exist, skip Steps 1 + 2 ---
        pipeline_state = cache.load_pipeline_state(cache_key)
        if pipeline_state is not None and cache.exists(cache_key):
            logger.info("=== Full pipeline cache hit — loading from disk (Steps 1 + 2 skipped) ===")
            track_histories = pipeline_state["track_histories"]
            metadata = pipeline_state["metadata"]
            embedding_matrix, frame_index_entries = cache.load(cache_key)
        else:
            logger.info(
                "Video: %dx%d @ %.1f fps, %.1f sec, %d frames (effective_skip=%d)",
                metadata.width, metadata.height, metadata.fps,
                metadata.duration_sec, metadata.total_frames, effective_skip,
            )

            detector = self._get_detector()
            tracker = self._get_tracker()
            frame_processor = FrameProcessor(detector, tracker)
            tracker.reset()

            # --- Steps 1+2 (streaming): detection + tracking + CLIP encoding in one pass.
            # Pixel data is dropped after each CLIP batch, so memory stays O(batch_size)
            # regardless of video length.
            t1 = time.time()
            logger.info(
                "Step 1+2/3 (streaming): YOLO+DeepSORT+CLIP (yolo=%s, clip=%s, cache_key=%s...) ...",
                self.config.yolo.model, self.config.clip.model_name, cache_key[:8],
            )
            embedding_builder.stream_start(
                batch_size=self.config.pipeline.batch_size,
                skip_static=self.config.pipeline.skip_static_frames,
                static_diff_threshold=self.config.pipeline.static_diff_threshold,
            )
            total_to_process = max(1, metadata.total_frames // effective_skip)
            processed = 0

            for frame_index, timestamp_sec, frame_bgr in tqdm(
                loader.iter_frames(frame_skip=effective_skip),
                total=total_to_process,
                desc="YOLO+DeepSORT+CLIP",
                unit="frame",
            ):
                pf = frame_processor.process(frame_bgr, frame_index, timestamp_sec)
                embedding_builder.stream_add(pf)
                processed += 1
                if progress_callback is not None and processed % 25 == 0:
                    progress_callback(
                        min(processed / total_to_process, 1.0),
                        f"Processing frame {processed:,} / ~{total_to_process:,}",
                    )

            embedding_matrix, frame_index_entries = embedding_builder.stream_finalize(cache_key)
            track_histories = tracker.get_track_histories()
            logger.info(
                "Step 1+2/3 done in %.1fs — %d frames processed, %d embedded, %d unique tracks",
                time.time() - t1, processed, embedding_matrix.shape[0], len(track_histories),
            )

            # --- Save pipeline state so next run of same video is instant ---
            cache.save_pipeline_state(cache_key, {
                "track_histories": track_histories,
                "metadata": metadata,
            })

        # --- Step 3: Run anomaly detection ---
        if progress_callback is not None:
            progress_callback(1.0, "Running anomaly detection …")
        t3 = time.time()
        logger.info("Step 3/3: Anomaly detection (rule_based=%s, vadclip=%s) ...",
                    self.config.anomaly.enable_rule_based, self.config.anomaly.enable_vadclip)
        anomaly_engine = AnomalyEngine(self.config)
        anomaly_events = anomaly_engine.analyze(
            track_histories=track_histories,
            fps=metadata.fps,
            frame_embeddings=embedding_matrix,
            frame_index_entries=frame_index_entries,
        )
        logger.info("Step 3/3 done in %.1fs — %d anomaly events", time.time() - t3, len(anomaly_events))
        for ev in anomaly_events:
            logger.info(
                "  Anomaly: type=%s track=%s t=%.1f–%.1fs severity=%.2f loc=(%.0f,%.0f)",
                ev.event_type, ev.track_id, ev.start_sec, ev.end_sec,
                ev.severity, ev.location_xy[0], ev.location_xy[1],
            )
        logger.info("=== Pipeline complete in %.1fs ===", time.time() - t_start)

        return PipelineResult(
            embedding_matrix=embedding_matrix,
            frame_index_entries=frame_index_entries,
            track_histories=track_histories,
            anomaly_events=anomaly_events,
            video_metadata=metadata,
        )
