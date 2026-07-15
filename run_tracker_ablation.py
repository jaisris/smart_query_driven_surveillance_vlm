"""DeepSORT vs ByteTrack ablation on the VIRAT 70s clip -> Docs/tracker_comparison.json"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")

import numpy as np
from utils.config_loader import get_config, clear_cache

VIDEO = "data/videos/VIRAT_S_000200_00_000100_000171.mp4"
results = {}

for backend in ["deepsort", "bytetrack"]:
    import shutil
    shutil.rmtree(".cache", ignore_errors=True)  # backend not in cache key - force fresh run
    clear_cache()
    cfg = get_config()
    cfg.tracking.backend = backend
    from pipeline.video_pipeline import VideoPipeline

    t0 = time.time()
    r = VideoPipeline(cfg).run(VIDEO)  # path-key cache; detection always re-runs? no —
    wall = time.time() - t0

    th = r.track_histories
    lens = [len(s) for s in th.values()]
    classes = {}
    for snaps in th.values():
        if snaps:
            classes[snaps[0].class_name] = classes.get(snaps[0].class_name, 0) + 1
    results[backend] = {
        "wall_sec": round(wall, 1),
        "unique_tracks": len(th),
        "mean_track_len_frames": round(float(np.mean(lens)), 1) if lens else 0,
        "max_track_len_frames": int(max(lens)) if lens else 0,
        "tracks_per_class": classes,
    }
    print(backend, results[backend])

out = {
    "video": VIDEO,
    "note": "Same 70s VIRAT clip, frame_skip=15 (141 frames), yolov8n, CPU. "
            "DeepSORT = separate YOLO detect + re-ID tracker; ByteTrack = single YOLO.track() pass.",
    "results": results,
}
os.makedirs("Docs", exist_ok=True)
with open("Docs/tracker_comparison.json", "w") as f:
    json.dump(out, f, indent=2)
print("SAVED -> Docs/tracker_comparison.json")
