"""1-hour video capacity test: streaming pipeline + adaptive skip + NL queries.

Uses a lower max_indexed_frames so the CPU test finishes in reasonable time;
memory behaviour (the thing under test) is identical at any setting.
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

VIDEO = r"C:\Users\MSUSERSL123\Desktop\ccsp\videos\DAY10.mp4"
QUERIES = ["a person presenting", "a slide with text on screen", "people in a meeting"]

cfg = get_config()
cfg.pipeline.max_indexed_frames = 800   # keep CPU runtime tractable for the test

print(f"Hashing {VIDEO} ...", flush=True)
h = hashlib.sha256()
with open(VIDEO, "rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
content_hash = h.hexdigest()

t0 = time.time()
result = VideoPipeline(cfg).run(VIDEO, content_hash=content_hash)
wall = time.time() - t0

out = {
    "video": VIDEO,
    "duration_sec": round(result.video_metadata.duration_sec, 1),
    "total_frames": result.video_metadata.total_frames,
    "frames_embedded": int(result.embedding_matrix.shape[0]),
    "unique_tracks": len(result.track_histories),
    "anomaly_events": len(result.anomaly_events),
    "pipeline_wall_sec": round(wall, 1),
    "retrieval": [],
}

search = SimilaritySearch(cfg)
search.build_index(result.embedding_matrix, result.frame_index_entries)
qenc = QueryEncoder(config=cfg)
for q in QUERIES:
    qt = time.time()
    segs = localize_segments(search.search(qenc.encode(q), top_k=20), min_score=0.20, config=cfg)
    out["retrieval"].append({
        "query": q,
        "latency_ms": round((time.time() - qt) * 1000, 1),
        "segments": [
            {"start": round(s.start_sec, 1), "end": round(s.end_sec, 1), "score": round(s.peak_score, 4)}
            for s in segs[:3]
        ],
    })

with open("capacity_test_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("SAVED -> capacity_test_results.json")
