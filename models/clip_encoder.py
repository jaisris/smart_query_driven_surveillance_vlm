"""Vision-language encoder for image frames and text queries.

Supports two model families via HuggingFace transformers:
  - CLIP   (e.g. openai/clip-vit-base-patch32, 512-d)
  - SigLIP / SigLIP 2 (e.g. google/siglip2-base-patch16-224, 768-d)

All outputs are L2-normalised float32 numpy arrays of shape (embedding_dim,).
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
        self.is_siglip = "siglip" in model_name.lower()

        logger.info("Loading %s model '%s' on %s ...",
                    "SigLIP" if self.is_siglip else "CLIP", model_name, self.device)
        if self.is_siglip:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        else:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Encoder loaded (embedding_dim=%d).", self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Output vector dimensionality (CLIP: projection_dim, SigLIP: hidden_size)."""
        cfg = self.model.config
        dim = getattr(cfg, "projection_dim", None)
        if not dim:
            dim = cfg.text_config.hidden_size
        return int(dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Encode a single RGB frame. Returns L2-normalised (embedding_dim,) vector."""
        return self.encode_image_batch([frame_rgb])[0]

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text query. Returns L2-normalised (embedding_dim,) vector."""
        import time
        t0 = time.time()
        prefix = self.config.clip.query_prefix
        full_query = f"{prefix} {query}" if prefix else query
        logger.info("encode_text: input='%s'", full_query)
        if self.is_siglip:
            # SigLIP was trained with fixed-length padded text; anything else
            # degrades its text embeddings badly.
            inputs = self.processor(
                text=[full_query], return_tensors="pt",
                padding="max_length", max_length=64, truncation=True,
            )
        else:
            inputs = self.processor(text=[full_query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        vec = features.cpu().numpy()[0]
        result = _l2_normalize(vec)
        logger.info("encode_text: output shape=%s, norm=%.4f, time=%.3fs",
                    result.shape, float(np.linalg.norm(result)), time.time() - t0)
        return result

    def encode_image_batch(self, frames_rgb: List[np.ndarray]) -> np.ndarray:
        """Encode a batch of RGB frames. Returns L2-normalised (N, embedding_dim) array."""
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
