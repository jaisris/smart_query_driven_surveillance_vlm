"""VadCLIP-based anomaly scoring.

VadCLIP (AAAI 2024) uses a dual-branch frozen CLIP for weakly-supervised
video anomaly detection — 88% AUC on UCF-Crime.

Reference: https://github.com/nwpu-zxr/VadCLIP

This wrapper:
  1. Takes the pre-computed CLIP frame embeddings (already available from
     the pipeline — no extra CLIP inference needed).
  2. Loads the VadCLIP MLP head weights trained on UCF-Crime.
  3. Returns a per-frame anomaly score in [0, 1].

Setup: download pretrained weights from the VadCLIP repo and set
  anomaly.vadclip_weights in config.yaml.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from utils.logger import get_logger
from utils.types import AnomalyEvent, FrameIndexEntry

logger = get_logger(__name__)

_ANOMALY_THRESHOLD = 0.5   # frames with score > this are flagged


class _VadCLIPHead(nn.Module):
    """Lightweight MLP head that VadCLIP trains on top of frozen CLIP features."""
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class VadCLIPDetector:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.model = _VadCLIPHead(input_dim=512).to(device)
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            # VadCLIP checkpoints may be wrapped; try common key variants
            state_dict = state.get("model_state_dict", state.get("state_dict", state))
            try:
                self.model.load_state_dict(state_dict)
                logger.info("VadCLIP weights loaded from %s", weights_path)
            except Exception as e:
                logger.warning("Could not load VadCLIP weights (%s). Using random init.", e)
        else:
            logger.warning("VadCLIP weights not found at '%s'. Using random init — scores will be meaningless.", weights_path)
        self.model.eval()

    def score_video(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """Return per-frame anomaly scores in [0, 1], shape (N,)."""
        x = torch.from_numpy(embedding_matrix.astype(np.float32)).to(self.device)
        with torch.no_grad():
            scores = self.model(x).cpu().numpy()
        return scores


def vadclip_to_anomaly_events(
    scores: np.ndarray,
    frame_index_entries: List[FrameIndexEntry],
    threshold: float = _ANOMALY_THRESHOLD,
    min_duration_sec: float = 1.0,
) -> List[AnomalyEvent]:
    """Convert per-frame scores into AnomalyEvent objects by thresholding and merging."""
    if len(scores) != len(frame_index_entries):
        raise ValueError("scores and frame_index_entries must have the same length")

    events: List[AnomalyEvent] = []
    in_event = False
    start_entry = None
    last_entry = None
    peak_score = 0.0

    for score, entry in zip(scores, frame_index_entries):
        if score > threshold:
            if not in_event:
                in_event = True
                start_entry = entry
                peak_score = score
            else:
                peak_score = max(peak_score, score)
            last_entry = entry
        else:
            if in_event:
                in_event = False
                dur = last_entry.timestamp_sec - start_entry.timestamp_sec
                if dur >= min_duration_sec or start_entry == last_entry:
                    events.append(AnomalyEvent(
                        track_id=-1,          # VadCLIP is not track-specific
                        event_type="vadclip_anomaly",
                        start_sec=start_entry.timestamp_sec,
                        end_sec=last_entry.timestamp_sec,
                        location_xy=(0.0, 0.0),
                        severity=round(float(peak_score), 3),
                    ))
                start_entry = None
                last_entry = None
                peak_score = 0.0

    # Close open event at end of video
    if in_event and start_entry is not None and last_entry is not None:
        events.append(AnomalyEvent(
            track_id=-1,
            event_type="vadclip_anomaly",
            start_sec=start_entry.timestamp_sec,
            end_sec=last_entry.timestamp_sec,
            location_xy=(0.0, 0.0),
            severity=round(float(peak_score), 3),
        ))

    return events
