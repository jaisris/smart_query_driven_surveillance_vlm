"""Top-level pipeline orchestrator.

Single entry point for the Streamlit UI and notebooks:
    result = VideoPipeline().run("path/to/video.mp4")
"""

from __future__ import annotations

import os
import time
from typing import Optional

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
from utils.types import PipelineResult, ProcessedFrame

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

    def run(self, video_path: str, content_hash: str | None = None) -> PipelineResult:
        """Process a full video and return PipelineResult.

        Pass content_hash (SHA256 of file bytes) to enable content-based caching so that
        re-uploading the same video skips Steps 1 and 2 entirely.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        t_start = time.time()
        logger.info("=== VideoPipeline.run START: '%s' ===", os.path.basename(video_path))

        encoder = self._get_encoder()
        cache = CacheManager(self.config.pipeline.cache_dir)
        embedding_builder = EmbeddingBuilder(encoder, cache)

        # --- Cache key: prefer content hash so re-uploads of the same file hit the cache ---
        if content_hash:
            cache_key = cache.content_key(
                content_hash,
                self.config.pipeline.frame_skip,
                self.config.clip.model_name,
            )
        else:
            cache_key = cache.key(
                video_path,
                self.config.pipeline.frame_skip,
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
            # --- Setup components ---
            loader = VideoLoader(video_path, self.config)
            metadata = loader.get_metadata()
            logger.info(
                "Video: %dx%d @ %.1f fps, %.1f sec, %d frames (frame_skip=%d)",
                metadata.width, metadata.height, metadata.fps,
                metadata.duration_sec, metadata.total_frames,
                self.config.pipeline.frame_skip,
            )

            detector = self._get_detector()
            tracker = self._get_tracker()
            frame_processor = FrameProcessor(detector, tracker)
            tracker.reset()

            # --- Step 1: Run detection + tracking on every frame ---
            t1 = time.time()
            logger.info("Step 1/3: YOLO detection + DeepSORT tracking (model=%s) ...", self.config.yolo.model)
            processed_frames: list[ProcessedFrame] = []
            total_to_process = metadata.total_frames // self.config.pipeline.frame_skip

            for frame_index, timestamp_sec, frame_bgr in tqdm(
                loader.iter_frames(),
                total=total_to_process,
                desc="YOLO+DeepSORT",
                unit="frame",
            ):
                pf = frame_processor.process(frame_bgr, frame_index, timestamp_sec)
                processed_frames.append(pf)

            track_histories = tracker.get_track_histories()
            logger.info(
                "Step 1/3 done in %.1fs — %d frames processed, %d unique tracks",
                time.time() - t1, len(processed_frames), len(track_histories),
            )

            # --- Step 2: Build CLIP embeddings (cache-first) ---
            t2 = time.time()
            logger.info("Step 2/3: CLIP embeddings (model=%s, cache_key=%s...) ...",
                        self.config.clip.model_name, cache_key[:8])
            embedding_matrix, frame_index_entries = embedding_builder.build_all_from_cache_or_encode(
                processed_frames,
                cache_key=cache_key,
                batch_size=self.config.pipeline.batch_size,
            )
            logger.info(
                "Step 2/3 done in %.1fs — embedding matrix shape %s",
                time.time() - t2, embedding_matrix.shape,
            )

            # --- Save pipeline state so next run of same video is instant ---
            cache.save_pipeline_state(cache_key, {
                "track_histories": track_histories,
                "metadata": metadata,
            })

        # --- Step 3: Run anomaly detection ---
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
