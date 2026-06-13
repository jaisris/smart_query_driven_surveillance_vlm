"""Anomaly detection evaluation: AUC-ROC and PR curve."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_auc_roc(y_true: List[int], y_scores: List[float]) -> float:
    """Frame-level AUC-ROC (standard metric for UCF-Crime)."""
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_scores))


def compute_ap_anomaly(y_true: List[int], y_scores: List[float]) -> float:
    """Average Precision for anomaly detection (area under PR curve)."""
    from sklearn.metrics import average_precision_score
    if len(set(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_scores))


def evaluate_anomaly_detection(
    y_true: List[int],
    y_scores: List[float],
) -> Dict[str, float]:
    """Full evaluation suite for anomaly detection.

    Args:
        y_true:   Frame-level binary labels (0=normal, 1=anomalous)
        y_scores: Per-frame anomaly scores in [0, 1]
    """
    from sklearn.metrics import roc_curve, precision_recall_curve

    auc = compute_auc_roc(y_true, y_scores)
    ap = compute_ap_anomaly(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)

    # Equal Error Rate
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    return {
        "AUC-ROC": round(auc, 4),
        "AP": round(ap, 4),
        "EER": round(eer, 4),
    }
