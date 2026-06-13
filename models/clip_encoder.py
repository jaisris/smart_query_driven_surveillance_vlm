"""CLIP encoder for image frames and text queries.

Uses HuggingFace transformers (not the openai/clip pip package).
All outputs are L2-normalised float32 numpy arrays of shape (512,).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


class CLIPEncoder:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        self.device = _resolve_device(self.config.clip.device)
        model_name = self.config.clip.model_name

        logger.info("Loading CLIP model '%s' on %s ...", model_name, self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Encode a single RGB frame. Returns L2-normalised (512,) vector."""
        return self.encode_image_batch([frame_rgb])[0]

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text query. Returns L2-normalised (512,) vector."""
        prefix = self.config.clip.query_prefix
        full_query = f"{prefix} {query}" if prefix else query
        inputs = self.processor(text=[full_query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        vec = features.cpu().numpy()[0]
        return _l2_normalize(vec)

    def encode_image_batch(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of RGB frames. Returns L2-normalised (N, 512) array."""
        pil_images = [Image.fromarray(f) for f in frames_rgb]
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        vecs = features.cpu().numpy()
        return _l2_normalize_batch(vecs)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _l2_normalize_batch(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)
