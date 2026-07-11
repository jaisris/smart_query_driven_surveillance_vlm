"""Ad-hoc evaluation runner: full pipeline + retrieval on a single video.

Produces genuine system-output values for the mid-term report:
detections, tracks, retrieval cosine scores/segments, anomaly events, latency.
"""
import hashlib
import json
import sys
import time

import numpy as np

from pipeline.video_pipeline import VideoPipeline
from retrieval.similarity_search import SimilaritySearch
from retrieval.temporal_localizer import localize_segments
from models.clip_encoder import CLIPEncoder
from utils.config_loader import get_config

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\MSUSERSL123\Downloads\videoplayback.mp4"
QUERIES = ["buses passing", "person walking", "cars colliding"]

cfg = get_config()
out = {"video": VIDEO, "config": {
    "frame_skip": cfg.pipeline.frame_skip,
    "yolo_model": cfg.yolo.model,
    "clip_model": cfg.clip.model_name,
    "top_k": cfg.retrieval.top_k,
}}

# Match the Streamlit UI: content-hash caching so a previously-processed
# video reuses cached detections/embeddings instead of recomputing.
with open(VIDEO, "rb") as _f:
    content_hash = hashlib.sha256(_f.read()).hexdigest()

t0 = time.time()
pipe = VideoPipeline()
result = pipe.run(VIDEO, content_hash=content_hash)
out["wall_time_sec"] = round(time.time() - t0, 1)

md = result.video_metadata
out["video_metadata"] = {
    "width": md.width, "height": md.height, "fps": md.fps,
    "duration_sec": round(md.duration_sec, 1), "total_frames": md.total_frames,
}
out["frames_embedded"] = int(result.embedding_matrix.shape[0])
out["embedding_dim"] = int(result.embedding_matrix.shape[1])

# ---- Detection / tracking summary from track histories ----
th = result.track_histories
track_lengths = [len(snaps) for snaps in th.values()]
class_counts = {}
for snaps in th.values():
    if snaps:
        cn = snaps[0].class_name
        class_counts[cn] = class_counts.get(cn, 0) + 1
out["tracking"] = {
    "unique_tracks": len(th),
    "mean_track_len_frames": round(float(np.mean(track_lengths)), 2) if track_lengths else 0,
    "max_track_len_frames": int(max(track_lengths)) if track_lengths else 0,
    "tracks_per_class": class_counts,
}

# ---- Anomaly events ----
out["anomaly"] = {
    "total_events": len(result.anomaly_events),
    "by_type": {},
    "events": [],
}
for ev in result.anomaly_events:
    out["anomaly"]["by_type"][ev.event_type] = out["anomaly"]["by_type"].get(ev.event_type, 0) + 1
    out["anomaly"]["events"].append({
        "type": ev.event_type, "track_id": ev.track_id,
        "start_sec": round(ev.start_sec, 1), "end_sec": round(ev.end_sec, 1),
        "severity": round(ev.severity, 2),
    })

# ---- Retrieval ----
search = SimilaritySearch(cfg)
search.build_index(result.embedding_matrix, result.frame_index_entries)
encoder = CLIPEncoder(cfg)
out["retrieval"] = []
for q in QUERIES:
    qt = time.time()
    qvec = encoder.encode_text(q)
    results = search.search(qvec, top_k=cfg.retrieval.top_k)
    segs = localize_segments(results, min_score=0.20, config=cfg)
    out["retrieval"].append({
        "query": q,
        "latency_ms": round((time.time() - qt) * 1000, 1),
        "n_results": len(results),
        "top_score": round(results[0].cosine_score, 4) if results else None,
        "top_timestamp_sec": round(results[0].timestamp_sec, 1) if results else None,
        "lowest_score": round(results[-1].cosine_score, 4) if results else None,
        "n_segments": len(segs),
        "segments": [{"start": round(s.start_sec, 1), "end": round(s.end_sec, 1),
                      "peak_score": round(s.peak_score, 4)} for s in segs[:5]],
    })

# ---- Throughput ----
if out["frames_embedded"]:
    out["throughput_fps"] = round(out["frames_embedded"] / out["wall_time_sec"], 3)

with open("results_videoplayback.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
print("\nSAVED -> results_videoplayback.json")
