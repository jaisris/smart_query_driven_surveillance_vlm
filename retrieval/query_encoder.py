"""Text query encoder — thin wrapper over CLIPEncoder.encode_text()."""

from __future__ import annotations

import numpy as np

from models.clip_encoder import CLIPEncoder
from utils.config_loader import AppConfig, get_config


class QueryEncoder:
    def __init__(self, encoder: CLIPEncoder | None = None, config: AppConfig | None = None):
        self.config = config or get_config()
        self.encoder = encoder or CLIPEncoder(self.config)

    def encode(self, query: str) -> np.ndarray:
        """Encode a natural language query. Returns L2-normalised (512,) vector."""
        return self.encoder.encode_text(query)
