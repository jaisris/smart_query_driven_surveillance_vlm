# Smart Query-Driven Surveillance VLM

M.Tech AI/ML dissertation project — BITS WILP S2-25_DISSERTATION-NSP4.

## What This System Does

End-to-end pipeline: takes a surveillance video + natural language query → returns timestamped video segments matching the query + flags suspicious activity.

## Architecture

```
Input Video
  → VideoLoader (OpenCV, adaptive frame skip — long videos raise the stride)
  → single streaming pass, O(batch) memory:
      FrameProcessor: YOLOv8 detections + DeepSORT tracks
      EmbeddingBuilder.stream_add(): static/degenerate-frame skip → CLIP batch encode (cached .npy)
  → FAISS IndexFlatIP (cosine similarity)
  ← text query → CLIPEncoder → query embedding
  → TemporalLocalizer → VideoSegments (start_sec, end_sec)
  → AnomalyEngine: rule-based (fast) + VadCLIP (accurate)
  → Streamlit UI (progress bar; accepts browser upload or local file path)
```

Long-video support: pixels are dropped after each CLIP batch, so hour-plus videos
process in constant memory (~200 MB). Verified on a 3.9-hour 1080p video.

## Module Map

| Path | Role |
|------|------|
| `utils/types.py` | All shared dataclasses |
| `utils/config_loader.py` | Loads `configs/config.yaml` → `AppConfig` |
| `data/video_loader.py` | Frame iterator over a video file |
| `data/cache_manager.py` | Save/load `.npy` embeddings + `index.json` |
| `models/clip_encoder.py` | CLIP **or SigLIP 2** image + text encoder (`clip.model_name`) |
| `models/yolo_detector.py` | YOLOv8 wrapper → `List[Detection]` (DeepSORT path only) |
| `models/deepsort_tracker.py` | DeepSORT baseline tracker (`tracking.backend: deepsort`) |
| `models/ultralytics_tracker.py` | ByteTrack/BoT-SORT combined detect+track (default backend) |
| `models/open_vocab_detector.py` | YOLO-World v2 — query terms become detection vocabulary |
| `pipeline/video_pipeline.py` | Top-level orchestrator → `PipelineResult` |
| `retrieval/similarity_search.py` | FAISS index build + query search |
| `retrieval/temporal_localizer.py` | Merge top-K frames into segments |
| `anomaly/anomaly_engine.py` | Aggregate loitering + intrusion + VadCLIP |
| `ui/app.py` | Streamlit demo app |
| `evaluation/` | mAP, MOTA/IDF1, AUC-ROC metrics |
| `evaluation/run_ucf_eval.py` | CLIP zero-shot AUC-ROC on the UCF-Crime frames dataset |
| `notebooks/04_colab_gpu_pipeline.ipynb` | Run pipeline + eval on Colab GPU; cache is portable |

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
- `pipeline.frame_skip`: reduce to 1 for max accuracy, increase for speed (default 15)
- `pipeline.max_indexed_frames`: cap on indexed frames; long videos auto-raise the skip (default 4000)
- `pipeline.skip_static_frames`: skip CLIP encoding of near-duplicate frames (default true)
- `tracking.backend`: `bytetrack` (default) | `botsort` | `deepsort` (baseline for ablation)
- `clip.model_name`: `openai/clip-vit-base-patch32` (default) | `google/siglip2-base-patch16-224`
- `retrieval.query_aware_detection`: YOLO-World boxes labelled with query terms on top result (default true)
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
- **Cache key**: SHA256 of `(file content hash or path, effective_frame_skip, clip_model_name)` — content-based, so re-uploads and cross-machine transfers hit the cache; delete `.cache/` to force re-encode
- **VadCLIP**: anomaly detection fallback; requires pretrained weights from https://github.com/nwpu-zxr/VadCLIP
