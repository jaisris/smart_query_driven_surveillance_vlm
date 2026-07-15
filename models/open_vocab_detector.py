"""Open-vocabulary detection with YOLO-World v2.

Detects arbitrary text-described objects ("person carrying a bag", "red car")
instead of the fixed 80 COCO classes. Used at display time: after retrieval
returns matching frames, the user's own query terms become the detection
vocabulary, so result thumbnails get boxes labelled with what was asked for.

Weights (yolov8s-worldv2.pt, ~50 MB) download automatically on first use.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from ultralytics import YOLOWorld

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import Detection

logger = get_logger(__name__)

# Words that carry no visual meaning on their own in a detection vocabulary
# (prepositions, verbs of motion, and scene words that yield whole-frame boxes).
_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "with", "near", "next",
    "to", "and", "or", "is", "are", "was", "were", "there", "some", "any",
    "show", "find", "me", "all", "video", "footage", "scene", "frame",
    "across", "through", "over", "under", "behind", "front", "beside",
    "walking", "running", "moving", "standing", "sitting", "going",
    "parking", "lot", "area", "zone", "street", "road", "entrance", "exit",
    "building", "background",
}


def query_to_vocabulary(query: str, max_terms: int = 4) -> List[str]:
    """Turn a natural-language query into open-vocab detection prompts.

    The full query is always the first prompt (YOLO-World handles phrases);
    individual content words are added as fallbacks so partial matches
    still get boxes (e.g. "person" when "person carrying a bag" misses).
    """
    query = query.strip().lower()
    if not query:
        return []
    vocab = [query]
    for word in re.findall(r"[a-z]+", query):
        if word not in _STOPWORDS and word not in vocab:
            vocab.append(word)
        if len(vocab) >= max_terms + 1:
            break
    return vocab


class OpenVocabDetector:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        model_name = self.config.yolo.open_vocab_model
        device_str = self.config.yolo.device
        if device_str == "auto":
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device_str

        logger.info("Loading YOLO-World model '%s' on %s ...", model_name, device_str)
        self._model = YOLOWorld(model_name)
        self._current_vocab: List[str] = []
        logger.info("YOLO-World loaded.")

    def detect(
        self,
        frame_bgr: np.ndarray,
        vocabulary: List[str],
        conf: float = 0.05,
    ) -> List[Detection]:
        """Detect the given text-described classes in one BGR frame.

        conf is intentionally low: open-vocab confidence scores run far below
        closed-set YOLO scores, especially for phrase prompts.
        """
        if not vocabulary:
            return []
        if vocabulary != self._current_vocab:
            self._model.set_classes(vocabulary)
            self._current_vocab = list(vocabulary)

        result = self._model.predict(
            frame_bgr, conf=conf, device=self._device, verbose=False
        )[0]

        detections: List[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                Detection(
                    bbox_xyxy=box.xyxy[0].tolist(),
                    class_id=cls_id,
                    class_name=result.names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                )
            )
        return detections
