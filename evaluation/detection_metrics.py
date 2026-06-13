"""Object detection evaluation: mAP@50 via IoU matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / (union + 1e-8)


def compute_ap(
    detections: List[Tuple[float, bool]],
    n_gt: int,
) -> float:
    """Compute Average Precision from a list of (confidence, is_tp) pairs."""
    if n_gt == 0:
        return 0.0
    detections = sorted(detections, key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for conf, is_tp in detections:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    # Interpolate using 11-point method
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = max([p for p, r in zip(precisions, recalls) if r >= t], default=0.0)
        ap += p / 11
    return ap


def compute_map(
    pred_boxes_per_image: List[List[Tuple[float, int, List[float]]]],
    gt_boxes_per_image: List[List[Tuple[int, List[float]]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mAP@iou_threshold across all classes.

    Args:
        pred_boxes_per_image: per image, list of (confidence, class_id, bbox_xyxy)
        gt_boxes_per_image:   per image, list of (class_id, bbox_xyxy)
        iou_threshold: IoU threshold for a detection to count as TP

    Returns:
        {"mAP": float, "per_class": {class_id: ap}}
    """
    # Group detections and GT by class
    class_dets: Dict[int, List[Tuple[float, bool]]] = {}
    class_gt_counts: Dict[int, int] = {}

    for img_preds, img_gts in zip(pred_boxes_per_image, gt_boxes_per_image):
        matched_gts = set()
        # Build GT lookup per class
        gt_by_class: Dict[int, List[Tuple[int, List[float]]]] = {}
        for gi, (cls, box) in enumerate(img_gts):
            gt_by_class.setdefault(cls, []).append((gi, box))
            class_gt_counts[cls] = class_gt_counts.get(cls, 0) + 1

        for conf, cls, pred_box in sorted(img_preds, key=lambda x: -x[0]):
            gts_this_class = gt_by_class.get(cls, [])
            best_iou, best_gi = 0.0, -1
            for gi, gt_box in gts_this_class:
                if gi in matched_gts:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi

            is_tp = best_iou >= iou_threshold and best_gi != -1
            if is_tp:
                matched_gts.add(best_gi)
            class_dets.setdefault(cls, []).append((conf, is_tp))

    per_class_ap = {}
    for cls, dets in class_dets.items():
        n_gt = class_gt_counts.get(cls, 0)
        per_class_ap[cls] = compute_ap(dets, n_gt)

    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {"mAP": mean_ap, "per_class": per_class_ap}
