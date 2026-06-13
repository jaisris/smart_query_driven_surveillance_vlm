# Smart Query-Driven Surveillance VLM

M.Tech AI/ML dissertation project — BITS WILP S2-25_DISSERTATION-NSP4.

## What This System Does

End-to-end pipeline: takes a surveillance video + natural language query → returns timestamped video segments matching the query + flags suspicious activity.

## Architecture

```
Input Video
  → VideoLoader (OpenCV, every Nth frame)
  → FrameProcessor: YOLOv8 detections + DeepSORT tracks
  → EmbeddingBuilder: CLIP frame embeddings (cached .npy)
  → FAISS IndexFlatIP (cosine similarity)
  ← text query → CLIPEncoder → query embedding
  → TemporalLocalizer → VideoSegments (start_sec, end_sec)
  → AnomalyEngine: rule-based (fast) + VadCLIP (accurate)
  → Streamlit UI
```

## Module Map

| Path | Role |
|------|------|
| `utils/types.py` | All shared dataclasses |
| `utils/config_loader.py` | Loads `configs/config.yaml` → `AppConfig` |
| `data/video_loader.py` | Frame iterator over a video file |
| `data/cache_manager.py` | Save/load `.npy` embeddings + `index.json` |
| `models/clip_encoder.py` | CLIP image + text encoder (HuggingFace transformers) |
| `models/yolo_detector.py` | YOLOv8 wrapper → `List[Detection]` |
| `models/deepsort_tracker.py` | DeepSORT wrapper → `List[Track]` + track histories |
| `pipeline/video_pipeline.py` | Top-level orchestrator → `PipelineResult` |
| `retrieval/similarity_search.py` | FAISS index build + query search |
| `retrieval/temporal_localizer.py` | Merge top-K frames into segments |
| `anomaly/anomaly_engine.py` | Aggregate loitering + intrusion + VadCLIP |
| `ui/app.py` | Streamlit demo app |
| `evaluation/` | mAP, MOTA/IDF1, AUC-ROC metrics |

## How to Run

```bash
pip install -r requirements.txt

# Run the demo UI
streamlit run ui/app.py

# Run the pipeline programmatically
python -c "
from pipeline.video_pipeline import VideoPipeline
result = VideoPipeline().run('path/to/video.mp4')
print(result.video_metadata)
"

# Run tests
pytest tests/ -v
```

## Config

All parameters live in `configs/config.yaml`. Key knobs:
- `pipeline.frame_skip`: reduce to 1 for max accuracy, increase to 10 for speed
- `yolo.model`: `yolov8n.pt` (fast) / `yolov8m.pt` (accurate)
- `anomaly.enable_vadclip`: set `true` after downloading VadCLIP weights
- `anomaly.intrusion.roi_zones`: add pixel polygons to enable intrusion detection

## Datasets

| Dataset | Path | Purpose |
|---------|------|---------|
| UCF-Crime | `data/datasets/UCF_Crimes/` | Anomaly detection baseline |
| UCA (UCF-Crime Annotation) | `data/datasets/UCA/` | NL temporal grounding eval |
| MOT17 | `data/datasets/MOT17/` | Tracking evaluation |

## Key Design Decisions

- **CLIP**: uses `transformers` library (not `openai/clip` pip package)
- **BGR→RGB**: conversion happens in `FrameProcessor`, never in `CLIPEncoder`
- **Bbox format**: always `[x1, y1, x2, y2]` absolute pixels throughout
- **Cache key**: SHA256 of `(video_path, frame_skip, clip_model_name)` — delete `.cache/` to force re-encode
- **VadCLIP**: anomaly detection fallback; requires pretrained weights from https://github.com/nwpu-zxr/VadCLIP
