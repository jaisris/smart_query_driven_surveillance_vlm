# Vision-Language Based Smart Surveillance System

> Query-Driven Video Retrieval and Suspicious Activity Detection using YOLOv8, DeepSORT, and CLIP

M.Tech AI & ML Dissertation — BITS WILP (S2-25_DISSERTATION-NSP4)
**Student:** Jaisri S (2024AA05138) | **Supervisor:** Savitha C, IBM

---

## Overview

This system lets users search long surveillance videos using plain English — no manual scrubbing, no keyword tags. Type a query like *"person running with a bag"* and the system retrieves the exact video segments that match, ranked by semantic similarity.

On top of retrieval, it continuously tracks all objects across frames and flags suspicious activities such as loitering and zone intrusion.

**Pipeline at a glance:**

```
Input Video
  └─ VideoLoader (OpenCV, frame-skip)
       └─ YOLOv8  →  object detections
       └─ DeepSORT →  multi-object tracks
       └─ CLIP     →  frame embeddings  ─────────┐
                                                  │
  Natural Language Query                          ▼
  └─ CLIP text encoder  →  query embedding  →  FAISS cosine search
                                                  │
                                                  ▼
                                         Temporal Localizer
                                         (merge top-K hits into segments)
                                                  │
                                    ┌─────────────┘
                                    ▼
                           Anomaly Engine
                           ├─ Rule-based: loitering, intrusion
                           └─ VadCLIP (AAAI 2024, 88% AUC on UCF-Crime)
                                                  │
                                                  ▼
                                        Streamlit Demo UI
```

---

## Features

- **Natural Language Video Retrieval** — query surveillance footage with free-form text; returns timestamped segments ranked by CLIP cosine similarity
- **Object Detection** — YOLOv8 detects persons, vehicles, and more in real time
- **Multi-Object Tracking** — DeepSORT assigns persistent IDs across frames
- **Temporal Localisation** — top-K matching frames are merged into contiguous video segments
- **Suspicious Activity Detection**
  - Rule-based: loitering (dwell time threshold), zone intrusion (polygon ROI)
  - VadCLIP: weakly-supervised CLIP-based anomaly scoring (AAAI 2024)
- **Embedding Cache** — CLIP frame embeddings are cached as `.npy` files; re-processing the same video is instant
- **Interactive UI** — Streamlit app for upload, query, and results visualisation

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Multi-Object Tracking | [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime) |
| Vision-Language Model | [CLIP via HuggingFace Transformers](https://huggingface.co/openai/clip-vit-base-patch32) |
| Similarity Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Anomaly Detection | [VadCLIP (AAAI 2024)](https://github.com/nwpu-zxr/VadCLIP) |
| UI | [Streamlit](https://streamlit.io) |
| Deep Learning | PyTorch 2.3.1 |
| Video I/O | OpenCV 4.10 |

---

## Project Structure

```
smart_query_driven_surveillance_vlm/
├── configs/
│   └── config.yaml             # All tunable parameters
├── utils/
│   ├── types.py                # Shared dataclasses (Detection, Track, SearchResult, ...)
│   ├── config_loader.py        # Loads config.yaml → AppConfig
│   └── logger.py               # Centralised logging
├── data/
│   ├── video_loader.py         # OpenCV frame iterator
│   ├── cache_manager.py        # .npy embedding cache with SHA-based invalidation
│   └── dataset_utils.py        # UCF-Crime / UCA dataset helpers
├── models/
│   ├── clip_encoder.py         # CLIP image + text encoder
│   ├── yolo_detector.py        # YOLOv8 wrapper → List[Detection]
│   └── deepsort_tracker.py     # DeepSORT wrapper → List[Track] + histories
├── pipeline/
│   ├── frame_processor.py      # YOLO + DeepSORT per-frame processing
│   ├── embedding_builder.py    # CLIP encoding with batch + cache support
│   └── video_pipeline.py       # Top-level orchestrator → PipelineResult
├── retrieval/
│   ├── query_encoder.py        # Text query → CLIP embedding
│   ├── similarity_search.py    # FAISS index build + cosine search
│   └── temporal_localizer.py   # Merge top-K frames into VideoSegments
├── anomaly/
│   ├── loitering_detector.py   # Rule-based dwell time detection
│   ├── intrusion_detector.py   # Polygon ROI zone crossing detection
│   ├── vadclip_detector.py     # VadCLIP AAAI 2024 anomaly scoring
│   └── anomaly_engine.py       # Aggregates all detectors
├── ui/
│   └── app.py                  # Streamlit demo app
├── evaluation/
│   ├── detection_metrics.py    # mAP@50
│   ├── tracking_metrics.py     # MOTA, IDF1
│   ├── retrieval_metrics.py    # Precision@K, Recall@K, NDCG, MRR
│   └── anomaly_metrics.py      # AUC-ROC, AP, EER
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_embedding_analysis.ipynb
│   └── 03_anomaly_detection_experiments.ipynb
├── tests/                      # 36 unit tests (no GPU / internet required)
├── conftest.py                 # pytest setup (Windows OpenMP fix)
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- Windows / Linux / macOS
- GPU optional (CUDA 12.1 compatible); CPU-only works out of the box

### 1. Clone and create virtual environment

```bash
git clone https://github.com/<your-username>/smart_query_driven_surveillance_vlm.git
cd smart_query_driven_surveillance_vlm

python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install --prefer-binary -r requirements.txt
```

### 3. Run the demo UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser, upload a surveillance video, and start querying.

---

## Usage

### Demo UI (Streamlit)

1. Upload a `.mp4` / `.avi` / `.mkv` video
2. Click **Run Pipeline** — YOLO + DeepSORT + CLIP runs automatically (embeddings are cached for the next run)
3. Type a natural language query, e.g.:
   - `person running near the exit`
   - `individual loitering by the entrance`
   - `car parked in restricted zone`
4. Adjust **Top-K** and **segment gap threshold** in the sidebar
5. View matching video segments with frame thumbnails, timestamps, and similarity scores
6. Scroll down to the **Anomaly Events** table for loitering / intrusion alerts

### Programmatic API

```python
from pipeline.video_pipeline import VideoPipeline
from retrieval.query_encoder import QueryEncoder
from retrieval.similarity_search import SimilaritySearch
from retrieval.temporal_localizer import localize_segments

# Run the full pipeline
result = VideoPipeline().run("path/to/surveillance.mp4")

# Query
encoder = QueryEncoder()
search = SimilaritySearch()
search.build_index(result.embedding_matrix, result.frame_index_entries)

query_vec = encoder.encode("person carrying a bag")
results = search.search(query_vec, top_k=10)
segments = localize_segments(results)

for seg in segments:
    print(f"{seg.start_sec:.1f}s – {seg.end_sec:.1f}s  (score: {seg.peak_score:.3f})")
```

---

## Configuration

All parameters are in [`configs/config.yaml`](configs/config.yaml):

| Key | Default | Description |
|-----|---------|-------------|
| `pipeline.frame_skip` | `5` | Process every Nth frame (~6 fps from 30 fps source) |
| `yolo.model` | `yolov8n.pt` | `yolov8n` (fast) / `yolov8m` (accurate) |
| `yolo.confidence_threshold` | `0.4` | Minimum detection confidence |
| `clip.model_name` | `openai/clip-vit-base-patch32` | CLIP variant |
| `retrieval.top_k` | `10` | Number of frames returned per query |
| `retrieval.gap_threshold_sec` | `2.0` | Max gap to merge adjacent results into one segment |
| `anomaly.loitering.dwell_time_sec` | `10.0` | Seconds before flagging loitering |
| `anomaly.loitering.dwell_radius_px` | `80` | Spatial radius for "staying in place" |
| `anomaly.intrusion.roi_zones` | `[]` | Pixel-coordinate polygons for forbidden zones |
| `anomaly.enable_vadclip` | `false` | Enable VadCLIP (requires pretrained weights) |

### VadCLIP setup (optional, for higher anomaly accuracy)

```bash
# Download weights from https://github.com/nwpu-zxr/VadCLIP
mkdir weights
# Place vadclip_ucf.pth in weights/
```

Then in `configs/config.yaml`:
```yaml
anomaly:
  enable_vadclip: true
  vadclip_weights: "weights/vadclip_ucf.pth"
```

---

## Datasets

| Dataset | Purpose | Link |
|---------|---------|------|
| UCF-Crime | Anomaly detection baseline (13 anomaly types, 1900 videos) | [CRCV UCF](https://www.crcv.ucf.edu/projects/real-world/) |
| UCA (UCF-Crime Annotation) | NL temporal grounding eval — 23,542 sentences with timestamps | [GitHub](https://github.com/Xuange923/Surveillance-Video-Understanding) |
| SurveillanceVQA-589K | VQA evaluation (2025 benchmark) | [HuggingFace](https://huggingface.co/datasets/fei213/SurveillanceVQA-589K) |
| MOT17 | Tracking evaluation (MOTA, IDF1) | [MOTChallenge](https://motchallenge.net/data/MOT17/) |
| COCO | YOLO pretrained weights (no download needed) | [cocodataset.org](https://cocodataset.org) |

Place datasets under `data/datasets/` and update paths in `configs/config.yaml`.

---

## Evaluation

```python
from evaluation.detection_metrics import compute_map
from evaluation.tracking_metrics import compute_tracking_metrics
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.anomaly_metrics import evaluate_anomaly_detection
```

| Task | Metric | Benchmark |
|------|--------|-----------|
| Object Detection | mAP@50 | COCO val |
| Tracking | MOTA, IDF1 | MOT17 |
| NL Retrieval | Precision@K, NDCG, MRR | UCA dataset |
| Anomaly Detection | AUC-ROC | UCF-Crime |

**SOTA reference on UCF-Crime (AUC-ROC):**
- VadCLIP (AAAI 2024): 88.02%
- π-VAD (CVPR 2025): 90.33%

---

## Tests

```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"
.\venv\Scripts\python.exe -m pytest tests/ -v

# Linux / macOS
pytest tests/ -v
```

36 tests covering: video loading, embedding cache, CLIP encoder, YOLO detector, FAISS similarity search, temporal localizer, and anomaly detection. No internet or GPU required.

---

## Related Work

- [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) — CLIP for end-to-end video clip retrieval
- [VadCLIP](https://github.com/nwpu-zxr/VadCLIP) — Weakly-supervised video anomaly detection with CLIP (AAAI 2024)
- [Towards Surveillance Video-and-Language Understanding](https://github.com/Xuange923/Surveillance-Video-Understanding) — CVPR 2024; closest academic work
- [YOLO-World](https://github.com/ailab-cvc/yolo-world) — Real-time open-vocabulary detection (future upgrade direction)

---

## License

Academic / research use only. Dataset usage subject to individual dataset licenses (UCF-Crime, MOT17, COCO).
