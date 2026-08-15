"""Run the full pipeline + a broad demo query set on a combined surveillance video.

Usage:
    python run_combined_demo.py <video_path> <output_json> <label>

Works unchanged on CPU (local) or GPU (Colab) - device is picked up from
configs/config.yaml (clip.device / yolo.device: auto).
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")

from pipeline.video_pipeline import VideoPipeline
from retrieval.query_encoder import QueryEncoder
from retrieval.similarity_search import SimilaritySearch
from retrieval.temporal_localizer import localize_segments
from utils.config_loader import get_config

VIDEO = sys.argv[1]
OUT_JSON = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(VIDEO)

# At the default max_indexed_frames=4000, a ~55-min combined video (~100k raw
# frames) forces effective_skip up from 15 to ~26 (see
# VideoPipeline._effective_frame_skip). That sparser sampling is enough to
# break DeepSORT/IOU track continuity on short (<30s) loitering-during-theft
# behavior that the same clip reliably caught standalone at skip=15 (see
# Docs/demo_realtheft_results.json). Raised only here (not in
# configs/config.yaml) so the dissertation's already-reported capacity/speed
# numbers at the default 4000 stay untouched; effective_skip is part of the
# cache key so this doesn't collide with caches built at the default.
config = get_config()
config.pipeline.max_indexed_frames = 8000

print(f"[{LABEL}] Hashing {VIDEO} ...", flush=True)
h = hashlib.sha256()
with open(VIDEO, "rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
content_hash = h.hexdigest()
print(f"[{LABEL}] content_hash:", content_hash[:16], flush=True)

t0 = time.time()
result = VideoPipeline(config).run(VIDEO, content_hash=content_hash)
wall = time.time() - t0
print(f"[{LABEL}] pipeline wall: {wall:.1f}s ({wall/60:.1f} min)", flush=True)

out = {
    "label": LABEL,
    "video": VIDEO,
    "duration_sec": round(result.video_metadata.duration_sec, 1),
    "duration_min": round(result.video_metadata.duration_sec / 60, 1),
    "frames_total": result.video_metadata.total_frames,
    "frames_indexed": int(result.embedding_matrix.shape[0]),
    "unique_tracks": len(result.track_histories),
    "pipeline_wall_sec": round(wall, 1),
    "pipeline_wall_min": round(wall / 60, 1),
    "anomaly_events": [
        {
            "track_id": e.track_id,
            "type": e.event_type,
            "start": round(e.start_sec, 1),
            "end": round(e.end_sec, 1),
            "severity": round(e.severity, 2),
            "loc": getattr(e, "location_xy", None),
        }
        for e in result.anomaly_events
    ],
    "queries": [],
}
print(f"[{LABEL}] tracks={out['unique_tracks']} frames_indexed={out['frames_indexed']} "
      f"anomaly_events={len(out['anomaly_events'])}", flush=True)

search = SimilaritySearch()
search.build_index(result.embedding_matrix, result.frame_index_entries)
qenc = QueryEncoder()

# Spans both general retrieval (should hit the MEVA/VIRAT backbone) and
# anomaly-specific queries (should hit the theft + loitering clips).
DEMO_QUERIES = [
    "people waiting at a bus stop",
    "a bus arriving",
    "a person walking alone",
    "a car driving past",
    "cars in a parking lot",
    "a car theft",
    "a person breaking into a car",
    "someone stealing a car",
    "a person loitering near parked cars",
    "a person lingering suspiciously",
    "suspicious activity near a vehicle",
]
for q in DEMO_QUERIES:
    segs = localize_segments(search.search(qenc.encode(q), top_k=20), min_score=0.20)
    out["queries"].append({
        "query": q,
        "top3": [{"start": round(s.start_sec, 1), "end": round(s.end_sec, 1),
                  "score": round(s.peak_score, 4)} for s in segs[:3]],
    })

os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"[{LABEL}] SAVED -> {OUT_JSON}")
