"""YOLOv8 object detection wrapper.

Returns Detection objects with bbox in [x1, y1, x2, y2] absolute pixels.
"""

from __future__ import annotations

from typing import List

import numpy as np
from ultralytics import YOLO

from utils.config_loader import AppConfig, get_config
from utils.logger import get_logger
from utils.types import Detection

logger = get_logger(__name__)

# COCO class names indexed by class_id
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant",
    # ... (abbreviated; YOLOv8 has all 80)
}


class YOLODetector:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or get_config()
        model_name = self.config.yolo.model
        device_str = self.config.yolo.device
        if device_str == "auto":
            import torch
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading YOLO model '%s' on %s ...", model_name, device_str)
        self._model = YOLO(model_name)
        self._device = device_str
        self._conf = self.config.yolo.confidence_threshold
        self._iou = self.config.yolo.iou_threshold
        self._whitelist = set(self.config.yolo.class_whitelist)
        logger.info("YOLO model loaded.")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a BGR frame. Returns detections for whitelisted classes."""
        results = self._model.predict(
            frame_bgr,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self._whitelist and cls_id not in self._whitelist:
                    continue
                xyxy = box.xyxy[0].tolist()      # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_name = result.names.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        bbox_xyxy=xyxy,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                    )
                )
        return detections
