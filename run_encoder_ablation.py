"""CLIP vs SigLIP2 retrieval comparison on the VIRAT 70s clip.

Runs the same queries against indexes built with each encoder and saves
Docs/encoder_retrieval_comparison.json. Cosine scores are not directly
comparable across encoders (different embedding spaces); what matters is
the RANKING and the retrieved timestamps.
"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")

from utils.config_loader import get_config, clear_cache

VIDEO = "data/videos/VIRAT_S_000200_00_000100_000171.mp4"
QUERIES = [
    "a person walking in a parking lot",
    "a white car parked",
    "a person getting out of a car",
]

out = {"video": VIDEO, "queries": QUERIES, "encoders": {}}

for model_name in ["openai/clip-vit-base-patch32", "google/siglip2-base-patch16-224"]:
    clear_cache()
    cfg = get_config()
    cfg.clip.model_name = model_name

    from pipeline.video_pipeline import VideoPipeline
    from retrieval.query_encoder import QueryEncoder
    from retrieval.similarity_search import SimilaritySearch
    from retrieval.temporal_localizer import localize_segments

    r = VideoPipeline(cfg).run(VIDEO)
    search = SimilaritySearch(cfg)
    search.build_index(r.embedding_matrix, r.frame_index_entries)
    qenc = QueryEncoder(config=cfg)

    entry = {"embedding_dim": int(r.embedding_matrix.shape[1]),
             "frames_indexed": int(r.embedding_matrix.shape[0]),
             "results": []}
    for q in QUERIES:
        results = search.search(qenc.encode(q), top_k=10)
        segs = localize_segments(results, min_score=0.0, config=cfg)
        entry["results"].append({
            "query": q,
            "top3_segments": [
                {"start": round(s.start_sec, 1), "end": round(s.end_sec, 1),
                 "peak_score": round(s.peak_score, 4)}
                for s in segs[:3]
            ],
        })
    out["encoders"][model_name] = entry
    print(model_name, "done")

with open("Docs/encoder_retrieval_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print("SAVED -> Docs/encoder_retrieval_comparison.json")
