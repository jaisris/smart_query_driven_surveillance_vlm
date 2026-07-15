"""CLIP zero-shot anomaly evaluation on the UCF-Crime frames dataset.

Dataset layout (Kaggle: odins0n/ucf-crime-dataset, extracted frames at 64x64):
    data/videos/archive/Test/<ClassName>/<VideoName>_x264_<frameNo>.png
    Classes: 13 anomaly types + NormalVideos

Scoring: each frame is CLIP-encoded and compared against two prompt banks
(normal scene descriptions vs. anomaly descriptions). The anomaly score is
    max(sim to anomaly prompts) - max(sim to normal prompts)
Frame-level and video-level AUC-ROC are reported (NormalVideos = 0, rest = 1).

Run:
    python evaluation/run_ucf_eval.py                     # 500 frames/class
    python evaluation/run_ucf_eval.py --frames-per-class 2000
    python evaluation/run_ucf_eval.py --data-root path/to/Test

Outputs:
    Docs/ucf_eval_results.json
    Docs/ucf_roc_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from evaluation.anomaly_metrics import evaluate_anomaly_detection
from models.clip_encoder import CLIPEncoder
from utils.config_loader import get_config
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Prompt banks for zero-shot scoring. Multiple phrasings per concept improve
# robustness (standard practice for CLIP zero-shot classification).
NORMAL_PROMPTS = [
    "a normal surveillance camera scene",
    "an empty street seen from a CCTV camera",
    "people walking calmly on a sidewalk",
    "normal traffic on a road",
    "a quiet shop interior from a security camera",
    "an ordinary parking lot with nothing happening",
]

ANOMALY_PROMPTS = [
    "people fighting violently",
    "a person attacking someone",
    "a building on fire with flames and smoke",
    "an explosion with a fireball",
    "a car crash on the road",
    "a person stealing from a store",
    "an armed robbery in progress",
    "a person breaking into a building",
    "a person firing a gun",
    "people vandalising property",
    "police arresting a person on the ground",
    "a person being abused or assaulted",
]

FRAME_RE = re.compile(r"^(?P<video>.+)_x264_(?P<frame>\d+)\.png$")


def collect_frames(data_root: str, frames_per_class: int, seed: int = 42) -> List[Tuple[str, str, int]]:
    """Return list of (image_path, video_id, label). Subsampled per class."""
    rng = random.Random(seed)
    samples: List[Tuple[str, str, int]] = []
    classes = sorted(os.listdir(data_root))
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        label = 0 if cls == "NormalVideos" else 1
        files = [f for f in os.listdir(cls_dir) if f.endswith(".png")]
        if len(files) > frames_per_class:
            files = rng.sample(files, frames_per_class)
        for fname in files:
            m = FRAME_RE.match(fname)
            video_id = f"{cls}/{m.group('video')}" if m else f"{cls}/{fname}"
            samples.append((os.path.join(cls_dir, fname), video_id, label))
        logger.info("Class %-15s: %5d frames sampled (label=%d)", cls, len(files), label)
    return samples


def encode_frames(encoder: CLIPEncoder, samples: List[Tuple[str, str, int]], batch_size: int = 64) -> np.ndarray:
    """CLIP-encode all sampled frames. Returns (N, 512) L2-normalised matrix."""
    from PIL import Image

    vecs: List[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        batch_paths = [p for p, _, _ in samples[start:start + batch_size]]
        images = [np.asarray(Image.open(p).convert("RGB")) for p in batch_paths]
        vecs.append(encoder.encode_image_batch(images))
        if (start // batch_size) % 10 == 0:
            done = start + len(batch_paths)
            rate = done / max(time.time() - t0, 1e-6)
            logger.info("Encoded %d/%d frames (%.0f frames/s)", done, len(samples), rate)
    return np.concatenate(vecs, axis=0)


def zero_shot_scores(encoder: CLIPEncoder, frame_vecs: np.ndarray) -> np.ndarray:
    """Anomaly score per frame: max anomaly-prompt sim minus max normal-prompt sim."""
    normal_mat = np.stack([encoder.encode_text(p) for p in NORMAL_PROMPTS])     # (P_n, 512)
    anomaly_mat = np.stack([encoder.encode_text(p) for p in ANOMALY_PROMPTS])   # (P_a, 512)
    sim_normal = frame_vecs @ normal_mat.T     # (N, P_n)
    sim_anomaly = frame_vecs @ anomaly_mat.T   # (N, P_a)
    return sim_anomaly.max(axis=1) - sim_normal.max(axis=1)


def video_level(samples: List[Tuple[str, str, int]], scores: np.ndarray) -> Tuple[List[int], List[float]]:
    """Aggregate frame scores per video (max pooling — standard for UCF-Crime)."""
    by_video: Dict[str, List[float]] = defaultdict(list)
    video_labels: Dict[str, int] = {}
    for (_, video_id, label), score in zip(samples, scores):
        by_video[video_id].append(float(score))
        video_labels[video_id] = label
    y_true = [video_labels[v] for v in by_video]
    y_score = [max(s) for s in by_video.values()]
    return y_true, y_score


def plot_roc(y_true: List[int], y_scores: List[float], auc: float, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"CLIP zero-shot (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1, ls="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("UCF-Crime Anomaly Detection — Frame-level ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("ROC curve saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/videos/archive/Test")
    parser.add_argument("--frames-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="Docs")
    parser.add_argument(
        "--model-name", default=None,
        help="Override clip.model_name (e.g. google/siglip2-base-patch16-224) "
             "for encoder ablations; output filenames get a model suffix.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise SystemExit(f"Dataset not found: {args.data_root}")
    os.makedirs(args.out_dir, exist_ok=True)

    config = get_config()
    if args.model_name:
        config.clip.model_name = args.model_name
    model_name = config.clip.model_name
    # e.g. "siglip2-base-patch16-224" — used to keep ablation outputs apart
    model_slug = model_name.split("/")[-1]
    is_baseline = model_name == "openai/clip-vit-base-patch32"

    t_start = time.time()
    samples = collect_frames(args.data_root, args.frames_per_class)
    labels = np.array([lbl for _, _, lbl in samples])
    logger.info("Total: %d frames (%d anomalous, %d normal) from %d classes",
                len(samples), int(labels.sum()), int((labels == 0).sum()),
                len(set(p.split(os.sep)[-2] for p, _, _ in samples)))

    encoder = CLIPEncoder(config)
    frame_vecs = encode_frames(encoder, samples, batch_size=args.batch_size)
    scores = zero_shot_scores(encoder, frame_vecs)

    frame_metrics = evaluate_anomaly_detection(labels.tolist(), scores.tolist())
    vid_true, vid_scores = video_level(samples, scores)
    video_metrics = evaluate_anomaly_detection(vid_true, vid_scores)

    logger.info("Frame-level:  %s", frame_metrics)
    logger.info("Video-level:  %s  (%d videos)", video_metrics, len(vid_true))

    suffix = "" if is_baseline else f"_{model_slug}"
    roc_path = os.path.join(args.out_dir, f"ucf_roc_curve{suffix}.png")
    plot_roc(labels.tolist(), scores.tolist(), frame_metrics["AUC-ROC"], roc_path)

    results = {
        "dataset": "UCF-Crime (Kaggle odins0n/ucf-crime-dataset, 64x64 extracted frames)",
        "method": f"Zero-shot ({model_name}), prompt-bank contrast scoring",
        "note": (
            "Frames are 64x64 which is far below the encoder's native input size; "
            "scores are a lower bound on what full-resolution frames would achieve."
        ),
        "frames_evaluated": len(samples),
        "frames_per_class_cap": args.frames_per_class,
        "videos_evaluated": len(vid_true),
        "frame_level": frame_metrics,
        "video_level": video_metrics,
        "normal_prompts": NORMAL_PROMPTS,
        "anomaly_prompts": ANOMALY_PROMPTS,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    out_json = os.path.join(args.out_dir, f"ucf_eval_results{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s (total runtime %.1fs)", out_json, results["runtime_sec"])

    print(f"\n=== UCF-Crime Zero-Shot Evaluation ({model_slug}) ===")
    print(f"Frames: {len(samples)}   Videos: {len(vid_true)}")
    print(f"Frame-level: {frame_metrics}")
    print(f"Video-level: {video_metrics}")


if __name__ == "__main__":
    main()
