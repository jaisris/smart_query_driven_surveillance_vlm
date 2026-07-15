"""Pre-index the 45-min MEVA presentation video (content-hash cached)."""
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

VIDEO = "data/videos/MEVA_bus_G340_45min.mp4"

print("Hashing (chunked)...", flush=True)
h = hashlib.sha256()
with open(VIDEO, "rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
content_hash = h.hexdigest()
print("content_hash:", content_hash[:16], flush=True)

t0 = time.time()
result = VideoPipeline().run(VIDEO, content_hash=content_hash)
wall = time.time() - t0

out = {
    "video": VIDEO,
    "duration_min": round(result.video_metadata.duration_sec / 60, 1),
    "frames_total": result.video_metadata.total_frames,
    "frames_indexed": int(result.embedding_matrix.shape[0]),
    "unique_tracks": len(result.track_histories),
    "anomaly_events": len(result.anomaly_events),
    "pipeline_wall_min": round(wall / 60, 1),
    "queries": [],
}

search = SimilaritySearch()
search.build_index(result.embedding_matrix, result.frame_index_entries)
qenc = QueryEncoder()
for q in ["people waiting at a bus stop", "a bus arriving", "a person walking alone",
          "a group of people talking", "a car driving past"]:
    segs = localize_segments(search.search(qenc.encode(q), top_k=20), min_score=0.20)
    out["queries"].append({
        "query": q,
        "top3": [{"start": round(s.start_sec, 1), "end": round(s.end_sec, 1),
                  "score": round(s.peak_score, 4)} for s in segs[:3]],
    })

with open("Docs/meva_demo_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("SAVED -> Docs/meva_demo_results.json")
