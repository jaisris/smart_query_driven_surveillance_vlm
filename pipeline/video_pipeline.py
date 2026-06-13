"""Top-level pipeline orchestrator.

Single entry point for the Streamlit UI and notebooks:
    result = VideoPipeline().run("path/to/video.mp4")
"""

from __future__ import annotations

import os
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
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        self._detector: Optional[YOLODetector] = None
        self._tracker: Optional[DeepSORTTracker] = None
        self._encoder: Optional[CLIPEncoder] = None

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

    def run(self, video_path: str) -> PipelineResult:
        """Process a full video and return PipelineResult."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info("=== VideoPipeline.run('%s') ===", os.path.basename(video_path))

        # --- Setup components ---
        loader = VideoLoader(video_path, self.config)
        metadata = loader.get_metadata()
        detector = self._get_detector()
        tracker = self._get_tracker()
        encoder = self._get_encoder()
        cache = CacheManager(self.config.pipeline.cache_dir)
        frame_processor = FrameProcessor(detector, tracker)
        embedding_builder = EmbeddingBuilder(encoder, cache)

        tracker.reset()

        # --- Cache key ---
        cache_key = cache.key(
            video_path,
            self.config.pipeline.frame_skip,
            self.config.clip.model_name,
        )

        # --- Step 1: Run detection + tracking on every frame ---
        logger.info("Step 1/3: Running detection + tracking ...")
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

        logger.info("Tracked %d frames, %d unique tracks.", len(processed_frames), len(tracker.get_track_histories()))

        # --- Step 2: Build CLIP embeddings (cache-first) ---
        logger.info("Step 2/3: Building CLIP embeddings ...")
        embedding_matrix, frame_index_entries = embedding_builder.build_all_from_cache_or_encode(
            processed_frames,
            cache_key=cache_key,
            batch_size=self.config.pipeline.batch_size,
        )

        # --- Step 3: Run anomaly detection ---
        logger.info("Step 3/3: Running anomaly detection ...")
        track_histories = tracker.get_track_histories()
        anomaly_engine = AnomalyEngine(self.config)
        anomaly_events = anomaly_engine.analyze(
            track_histories=track_histories,
            fps=metadata.fps,
            frame_embeddings=embedding_matrix,
            frame_index_entries=frame_index_entries,
        )
        logger.info("Detected %d anomaly events.", len(anomaly_events))

        return PipelineResult(
            embedding_matrix=embedding_matrix,
            frame_index_entries=frame_index_entries,
            track_histories=track_histories,
            anomaly_events=anomaly_events,
            video_metadata=metadata,
        )
