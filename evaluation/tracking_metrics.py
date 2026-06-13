"""Multi-object tracking evaluation: MOTA, IDF1 via motmetrics."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_tracking_metrics(
    pred_tracks: Dict[int, List[Tuple[int, List[float]]]],
    gt_tracks: Dict[int, List[Tuple[int, List[float]]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute MOTA and IDF1 using motmetrics.

    Args:
        pred_tracks: {track_id: [(frame_index, bbox_xyxy), ...]}
        gt_tracks:   {gt_id: [(frame_index, bbox_xyxy), ...]}
        iou_threshold: IoU threshold for matching

    Returns:
        {"MOTA": float, "IDF1": float, "MT": int, "ML": int, "FP": int, "FN": int, "IDs": int}
    """
    try:
        import motmetrics as mm
    except ImportError:
        return {"error": "motmetrics not installed. Run: pip install motmetrics"}

    acc = mm.MOTAccumulator(auto_id=True)

    # Collect all frame indices
    all_frames = sorted(
        set(fi for snaps in gt_tracks.values() for fi, _ in snaps)
        | set(fi for snaps in pred_tracks.values() for fi, _ in snaps)
    )

    # Build per-frame lookup
    gt_by_frame: Dict[int, Dict[int, List[float]]] = {}
    for gt_id, snaps in gt_tracks.items():
        for fi, box in snaps:
            gt_by_frame.setdefault(fi, {})[gt_id] = box

    pred_by_frame: Dict[int, Dict[int, List[float]]] = {}
    for track_id, snaps in pred_tracks.items():
        for fi, box in snaps:
            pred_by_frame.setdefault(fi, {})[track_id] = box

    from evaluation.detection_metrics import compute_iou

    for frame_idx in all_frames:
        gt_ids = list(gt_by_frame.get(frame_idx, {}).keys())
        pred_ids = list(pred_by_frame.get(frame_idx, {}).keys())
        gt_boxes = [gt_by_frame[frame_idx][i] for i in gt_ids]
        pred_boxes = [pred_by_frame[frame_idx][i] for i in pred_ids]

        dist_matrix = np.full((len(gt_ids), len(pred_ids)), np.inf)
        for gi, gb in enumerate(gt_boxes):
            for pi, pb in enumerate(pred_boxes):
                iou = compute_iou(gb, pb)
                if iou >= iou_threshold:
                    dist_matrix[gi, pi] = 1.0 - iou

        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "idf1", "num_misses", "num_false_positives", "num_switches", "mostly_tracked", "mostly_lost"], name="overall")

    row = summary.iloc[0]
    return {
        "MOTA": round(float(row.get("mota", 0.0)), 4),
        "IDF1": round(float(row.get("idf1", 0.0)), 4),
        "FN": int(row.get("num_misses", 0)),
        "FP": int(row.get("num_false_positives", 0)),
        "IDs": int(row.get("num_switches", 0)),
        "MT": int(row.get("mostly_tracked", 0)),
        "ML": int(row.get("mostly_lost", 0)),
    }
