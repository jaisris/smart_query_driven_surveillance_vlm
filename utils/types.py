"""Shared dataclasses used across all modules."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: List[float]      # [x1, y1, x2, y2] absolute pixels
    class_id: int
    class_name: str
    confidence: float

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bbox_tlwh(self) -> List[float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]


@dataclass
class TrackSnapshot:
    frame_index: int
    timestamp_sec: float
    centroid_xy: Tuple[float, float]
    bbox_xyxy: List[float]
    class_name: str


@dataclass
class Track:
    track_id: int
    bbox_xyxy: List[float]      # [x1, y1, x2, y2]
    class_name: str
    age_frames: int
    is_confirmed: bool

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class VideoMetadata:
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int
    path: str


@dataclass
class ProcessedFrame:
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    frame_index: int
    timestamp_sec: float
    detections: List[Detection]
    tracks: List[Track]


@dataclass
class FrameIndexEntry:
    frame_index: int
    timestamp_sec: float
    track_ids: List[int]


@dataclass
class FrameEmbedding:
    frame_index: int
    timestamp_sec: float
    vector: np.ndarray          # shape (512,), L2-normalised


@dataclass
class PipelineResult:
    embedding_matrix: np.ndarray            # shape (N, 512)
    frame_index_entries: List[FrameIndexEntry]
    track_histories: Dict[int, List[TrackSnapshot]]
    anomaly_events: List["AnomalyEvent"]
    video_metadata: VideoMetadata


@dataclass
class SearchResult:
    frame_index: int
    timestamp_sec: float
    cosine_score: float


@dataclass
class VideoSegment:
    start_sec: float
    end_sec: float
    peak_score: float
    frame_indices: List[int]

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def midpoint_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2


@dataclass
class AnomalyEvent:
    track_id: int
    event_type: str             # "loitering" | "intrusion"
    start_sec: float
    end_sec: float
    location_xy: Tuple[float, float]
    severity: float = 1.0       # ratio of observed dwell vs threshold (loitering) or 1.0 (intrusion)
