"""Streamlit UI — Smart Query-Driven Surveillance System."""

from __future__ import annotations

# Must be set before any DLL loads (fixes PyTorch + FAISS OpenMP conflict on Windows)
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
import logging
import os
import sys
import tempfile
import time
import traceback
from typing import List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import setup_logging
setup_logging(log_dir=os.path.join(os.path.dirname(__file__), "..", "logs"))

from pipeline.video_pipeline import VideoPipeline
from retrieval.query_encoder import QueryEncoder
from retrieval.similarity_search import SimilaritySearch
from retrieval.temporal_localizer import localize_segments
from utils.config_loader import get_config
from utils.types import AnomalyEvent, PipelineResult, VideoSegment

# ------------------------------------------------------------------ #
#  Page config  (must be the first Streamlit call)
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Smart Surveillance — Query-Driven Video Retrieval",
    page_icon="🎥",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  CSS
# ------------------------------------------------------------------ #

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════
   GLOBAL LIGHT THEME — overrides browser dark cache
   ═══════════════════════════════════════════════════ */

/* ── Page & sidebar backgrounds ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.main .block-container {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
section[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

/* ── Force dark text on all native elements ── */
.stApp p, .stApp span, .stApp div,
.stApp label, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp li,
.stApp small, .stApp a { color: #1e293b; }

/* ── Widget labels ── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span { color: #1e293b !important; }

/* ── Sliders ── */
[data-testid="stSlider"] span,
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] { color: #1e293b !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #3b82f6 !important; }

/* ── Checkboxes ── */
[data-testid="stCheckbox"] span,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] p { color: #1e293b !important; }

/* ── Expander — white bg, no pink ── */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #1e293b !important; background-color: #ffffff !important; }
[data-testid="stExpanderDetails"],
[data-testid="stExpanderDetails"] > div { background-color: #ffffff !important; color: #1e293b !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background-color: #f1f5f9 !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 10px !important;
    color: #475569 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small { color: #475569 !important; }

/* ── Buttons: primary = blue ── */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #3b82f6 !important;
    border-color: #3b82f6 !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover { background-color: #2563eb !important; border-color: #2563eb !important; }

/* ── Alert boxes ── */
[data-testid="stInfo"]    { background-color: #eff6ff !important; }
[data-testid="stInfo"] *  { color: #1e40af !important; }
[data-testid="stSuccess"] { background-color: #f0fdf4 !important; }
[data-testid="stSuccess"] * { color: #166534 !important; }
[data-testid="stError"]   { background-color: #fef2f2 !important; }
[data-testid="stError"] * { color: #991b1b !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #64748b !important; }

/* ═══════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ═══════════════════════════════════════════════════ */

.step-label {
    font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:1.2px; color:#3b82f6 !important; margin-bottom:2px;
}
.step-title {
    font-size:20px; font-weight:700; color:#1e293b !important;
    margin:0 0 16px 0; line-height:1.2;
}

.metric-row { display:flex; gap:12px; flex-wrap:wrap; }
.mc { background:#f1f5f9; border-radius:10px; padding:14px 18px; flex:1; min-width:90px; text-align:center; }
.mc-label { font-size:11px; font-weight:600; color:#64748b !important; text-transform:uppercase; letter-spacing:0.5px; }
.mc-value { font-size:24px; font-weight:700; color:#1e293b !important; margin-top:4px; }
.mc-sub   { font-size:12px; color:#94a3b8 !important; margin-top:2px; }

.seg-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }
.seg-title  { font-size:15px; font-weight:600; color:#1e293b !important; }
.seg-badge  { background:#dbeafe; color:#1d4ed8 !important; font-size:12px; font-weight:700; padding:3px 10px; border-radius:20px; }
.seg-meta   { font-size:13px; color:#64748b !important; margin-bottom:8px; }
.score-bar  { background:#e2e8f0; border-radius:4px; height:6px; overflow:hidden; margin-bottom:14px; }
.score-fill { height:6px; border-radius:4px; background:linear-gradient(90deg,#3b82f6,#06b6d4); }

.anom-card { display:flex; align-items:center; gap:16px; background:#fff; border-radius:10px;
    padding:16px 20px; margin-bottom:10px; border:1px solid #e2e8f0; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.anom-info  { flex:1; }
.anom-type  { font-size:14px; font-weight:700; color:#1e293b !important; text-transform:uppercase; letter-spacing:0.3px; }
.anom-meta  { font-size:13px; color:#64748b !important; margin-top:3px; }
.sev-H { background:#fee2e2; color:#b91c1c !important; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }
.sev-M { background:#fef3c7; color:#b45309 !important; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }
.sev-L { background:#dcfce7; color:#166534 !important; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Log capture
# ------------------------------------------------------------------ #

class _ListHandler(logging.Handler):
    def __init__(self, records: list):
        super().__init__()
        self._records = records
        self.setFormatter(logging.Formatter("%(levelname)s  %(name)s — %(message)s"))

    def emit(self, record: logging.LogRecord):
        self._records.append(self.format(record))


# ------------------------------------------------------------------ #
#  Session state
# ------------------------------------------------------------------ #

def _init_state():
    defaults = {
        "pipeline_result": None,
        "search_results": None,
        "video_segments": None,
        "video_path": None,
        "pipeline_logs": [],
        "pipeline_time_sec": None,
        "pipeline_from_cache": False,
        "query_text": "",
        "timeline": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

_app_logger = logging.getLogger("ui.app")

# ------------------------------------------------------------------ #
#  Cached models — load once per app session, reused across all runs
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading YOLO model …")
def _cached_yolo() -> "YOLODetector | None":
    # Combined tracking backends (bytetrack/botsort) run detection inside the
    # tracker; a separate YOLO detector is only used for the DeepSORT baseline.
    if get_config().tracking.backend != "deepsort":
        return None
    from models.yolo_detector import YOLODetector
    return YOLODetector(get_config())


@st.cache_resource(show_spinner="Loading CLIP model …")
def _cached_clip() -> "CLIPEncoder":
    from models.clip_encoder import CLIPEncoder
    return CLIPEncoder(get_config())


@st.cache_resource(show_spinner=False)
def _cached_tracker():
    cfg = get_config()
    if cfg.tracking.backend == "deepsort":
        from models.deepsort_tracker import DeepSORTTracker
        return DeepSORTTracker(cfg)
    from models.ultralytics_tracker import UltralyticsTracker
    return UltralyticsTracker(cfg)


@st.cache_resource(show_spinner=False)
def _cached_query_encoder() -> QueryEncoder:
    """Reuses the already-cached CLIP model — no second model load."""
    return QueryEncoder(encoder=_cached_clip(), config=get_config())


@st.cache_resource(show_spinner="Loading YOLO-World (open-vocabulary) …")
def _cached_open_vocab():
    """YOLO-World detector for query-aware boxes; None if unavailable."""
    try:
        from models.open_vocab_detector import OpenVocabDetector
        return OpenVocabDetector(get_config())
    except Exception as exc:
        logging.getLogger(__name__).warning("YOLO-World unavailable: %s", exc)
        return None


def _tail_log(n: int = 40) -> str:
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs", "surveillance.log")
    log_path = os.path.normpath(log_path)
    if not os.path.exists(log_path):
        return "(no log file found)"
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


# Colors are in BGR (OpenCV native). Frame is drawn in BGR then converted to RGB for display.
_CLASS_COLORS = {
    "person":     ( 80, 200,   0),   # BGR → green in display
    "bus":        (  0, 140, 255),   # BGR → orange in display
    "truck":      (220,  60, 180),   # BGR → purple in display
    "car":        (  0,   0, 220),   # BGR → red in display
    "motorcycle": (200, 200,   0),   # BGR → cyan in display
}
_CLASS_THICKNESS = {
    "bus": 3, "truck": 2, "person": 2, "motorcycle": 2, "car": 2,
}


def _draw_tracks(
    frame_bgr: np.ndarray,
    frame_idx: int,
    result: PipelineResult,
    highlight_classes: list | None = None,
    highlight_track_ids: set | None = None,
):
    """Draw bounding boxes with three levels:
    - highlight_track_ids: full colour + label (the specific tracks of interest)
    - target class but not in highlight_track_ids: thin grey, no label
    - wrong class: very faint grey, no label
    """
    for track_id, snapshots in result.track_histories.items():
        for snap in snapshots:
            if snap.frame_index != frame_idx:
                continue
            cls = snap.class_name.lower()
            in_target_class = (highlight_classes is None) or (cls in highlight_classes)
            in_target_track = (highlight_track_ids is None) or (track_id in highlight_track_ids)

            x1, y1, x2, y2 = [int(v) for v in snap.bbox_xyxy]

            if not in_target_class:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (180, 180, 180), 1)
                continue

            if not in_target_track:
                # Right class, not the specific track — muted, no label
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (160, 160, 160), 1)
                continue

            # Target track — full highlight + label
            color     = _CLASS_COLORS.get(cls, (200, 200, 200))
            thickness = _CLASS_THICKNESS.get(cls, 1)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame_bgr, f"{cls[:3].upper()}:{track_id}",
                (x1, max(y1 - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )


_OPEN_VOCAB_COLOR = (255, 0, 255)   # magenta (BGR) — visually distinct from class colours


def _draw_open_vocab(frame_bgr: np.ndarray, detections: list) -> None:
    """Draw YOLO-World boxes labelled with the user's own query terms."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        # Ignore near-whole-frame boxes (scene-level matches carry no localisation info)
        h, w = frame_bgr.shape[:2]
        if (x2 - x1) * (y2 - y1) > 0.6 * w * h:
            continue
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _OPEN_VOCAB_COLOR, 2)
        label = f"{det.class_name[:24]} {det.confidence:.2f}"
        cv2.putText(
            frame_bgr, label, (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _OPEN_VOCAB_COLOR, 1,
        )


_CLASS_KEYWORDS = {
    "person":     ["person", "people", "pedestrian", "man", "woman", "individual",
                   "someone", "anyone", "human", "suspect", "guard", "officer",
                   "running", "walking", "loitering", "sitting", "standing"],
    "bus":        ["bus", "buses", "coach", "minibus"],
    "car":        ["car", "cars", "vehicle", "sedan", "suv", "taxi", "cab",
                   "red car", "blue car", "white car", "black car", "silver car",
                   "collision", "crash", "accident", "colliding", "collide", "hit"],
    "truck":      ["truck", "trucks", "lorry", "van", "delivery"],
    "motorcycle": ["motorcycle", "bike", "scooter", "motorbike", "moped"],
}

_COLLISION_KEYWORDS = {"collision", "crash", "accident", "colliding", "collide", "hit"}

# BGR color masks for car colour detection (frame is in BGR from cv2)
_COLOR_MASKS_BGR = {
    "red":    lambda b, g, r: (r > 120) & (r > g * 1.4) & (r > b * 1.4),
    "blue":   lambda b, g, r: (b > 100) & (b > r * 1.4) & (b > g * 1.3),
    "white":  lambda b, g, r: (r > 190) & (g > 190) & (b > 190),
    "black":  lambda b, g, r: (r < 60)  & (g < 60)  & (b < 60),
    "yellow": lambda b, g, r: (r > 150) & (g > 150) & (b < 80),
    "green":  lambda b, g, r: (g > 100) & (g > r * 1.4) & (g > b * 1.3),
    "silver": lambda b, g, r: (r > 140) & (g > 140) & (b > 140),
}


def _query_highlight_classes(query: str) -> list | None:
    """Return the class(es) to highlight based on keywords in the query, or None for all."""
    q = query.lower()
    matches = []
    for cls, keywords in _CLASS_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matches.append(cls)
    return matches if matches else None


def _color_car_track_ids(
    frame_bgr: np.ndarray, frame_idx: int, result: PipelineResult,
    color_name: str, min_fraction: float = 0.15,
) -> set | None:
    """Return IDs of cars whose bounding box contains >= min_fraction pixels of target colour."""
    mask_fn = _COLOR_MASKS_BGR.get(color_name)
    if mask_fn is None:
        return None
    h, w = frame_bgr.shape[:2]
    matched: set = set()
    for track_id, snapshots in result.track_histories.items():
        for snap in snapshots:
            if snap.frame_index != frame_idx or snap.class_name.lower() != "car":
                continue
            x1, y1 = max(0, int(snap.bbox_xyxy[0])), max(0, int(snap.bbox_xyxy[1]))
            x2, y2 = min(w, int(snap.bbox_xyxy[2])), min(h, int(snap.bbox_xyxy[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame_bgr[y1:y2, x1:x2]
            b = roi[:, :, 0].astype(float)
            g = roi[:, :, 1].astype(float)
            r = roi[:, :, 2].astype(float)
            if np.sum(mask_fn(b, g, r)) / roi.size * 3 >= min_fraction:
                matched.add(track_id)
    return matched if matched else None


def _collision_track_ids(frame_idx: int, result: PipelineResult, max_dist_px: int = 100) -> set | None:
    """Return IDs of cars that overlap or are within max_dist_px of another car."""
    import math
    car_snaps = []
    for track_id, snapshots in result.track_histories.items():
        for snap in snapshots:
            if snap.frame_index == frame_idx and snap.class_name.lower() == "car":
                car_snaps.append((track_id, snap))

    collision_ids: set = set()
    for i, (tid1, snap1) in enumerate(car_snaps):
        x1a, y1a, x2a, y2a = snap1.bbox_xyxy
        for tid2, snap2 in car_snaps[i + 1:]:
            x1b, y1b, x2b, y2b = snap2.bbox_xyxy
            # Bounding box overlap
            if min(x2a, x2b) > max(x1a, x1b) and min(y2a, y2b) > max(y1a, y1b):
                collision_ids.update([tid1, tid2])
                continue
            # Centroid proximity
            cx1, cy1 = snap1.centroid_xy
            cx2, cy2 = snap2.centroid_xy
            if math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < max_dist_px:
                collision_ids.update([tid1, tid2])
    return collision_ids if collision_ids else None


QUERY_SUGGESTIONS = [
    "person loitering",
    "running near exit",
    "suspicious bag",
    "group gathering",
    "person near vehicle",
    "individual sitting alone",
]


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #

with st.sidebar:
    pipeline_done = st.session_state.pipeline_result is not None
    search_done   = st.session_state.video_segments is not None

    def _step_row(num: int, label: str, done: bool, active: bool) -> str:
        if done:
            icon, color, weight = "✅", "#16a34a", "600"
        elif active:
            icon, color, weight = "▶", "#3b82f6", "700"
        else:
            icon, color, weight = "○", "#94a3b8", "400"
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
            f'font-size:14px;font-weight:{weight};color:{color};">'
            f'{icon}&nbsp;{num}. {label}</div>'
        )

    st.markdown(
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1px;color:#64748b;margin-bottom:6px;">WORKFLOW</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_step_row(1, "Upload & Index", pipeline_done, not pipeline_done), unsafe_allow_html=True)
    st.markdown(_step_row(2, "Query Video",    search_done,   pipeline_done and not search_done), unsafe_allow_html=True)
    st.markdown(_step_row(3, "Review Results", False,         search_done), unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1px;color:#64748b;margin-bottom:8px;">SEARCH PARAMS</div>',
        unsafe_allow_html=True,
    )
    top_k         = st.slider("Top-K results", 1, 50, 20)
    min_score     = st.slider("Min score threshold", 0.0, 0.5, 0.20, 0.01)
    gap_threshold = st.slider("Segment gap (sec)", 0.5, 10.0, 2.0, 0.5)
    min_seg_dur   = st.slider("Min duration (sec)", 0.0, 5.0, 1.0, 0.5)

    st.divider()

    st.markdown(
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1px;color:#64748b;margin-bottom:8px;">ANOMALY DETECTION</div>',
        unsafe_allow_html=True,
    )
    enable_rule_based = st.checkbox("Rule-based (loitering / intrusion)", value=True)
    st.caption("VadCLIP requires pretrained weights — see CLAUDE.md.")

    st.divider()
    show_logs = st.checkbox("Show pipeline logs", value=False)

    st.divider()
    st.markdown(
        '<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        'letter-spacing:1px;color:#64748b;margin-bottom:6px;">BACKEND LOG</div>',
        unsafe_allow_html=True,
    )
    if st.button("Refresh log", use_container_width=True):
        st.session_state["_log_refresh"] = True
    with st.expander("Last 40 lines", expanded=False):
        st.code(_tail_log(40), language="text")


# ------------------------------------------------------------------ #
#  Page header
# ------------------------------------------------------------------ #

st.markdown(
    '<h1 style="color:#1e293b;font-size:30px;font-weight:800;margin-bottom:2px;">'
    '🎥&nbsp; Smart Query-Driven Surveillance System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#64748b;font-size:14px;margin-bottom:28px;">'
    'Vision-Language Video Retrieval + Suspicious Activity Detection &nbsp;·&nbsp; '
    'M.Tech Dissertation · BITS WILP 2026</p>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ #
#  STEP 1 — Upload & Pipeline
# ------------------------------------------------------------------ #

with st.container(border=True):
    st.markdown('<div class="step-label">STEP 1</div>', unsafe_allow_html=True)

    title_col, help_col = st.columns([5, 1])
    with title_col:
        st.markdown('<div class="step-title">Upload & Index Video</div>', unsafe_allow_html=True)
    with help_col:
        with st.expander("? Help"):
            st.markdown(
                "**How it works:**\n\n"
                "1. Drop any `.mp4 / .avi / .mkv / .mov` file into the uploader below.\n"
                "2. Click **▶ Run Pipeline** — the system will run YOLO object detection, "
                "DeepSORT tracking, and CLIP frame embeddings.\n"
                "3. Once complete, video stats appear and you can move to Step 2 to query the footage."
            )

    uploaded = st.file_uploader(
        "Drop a surveillance video",
        type=["mp4", "avi", "mkv", "mov", "mpeg4"],
        label_visibility="collapsed",
    )

    local_path = st.text_input(
        "…or path to a local video file (avoids uploading large files through the browser)",
        key="local_video_path",
        placeholder=r"e.g.  C:\videos\camera01_full_day.mp4",
    )
    _local_path_valid = bool(local_path.strip()) and os.path.isfile(local_path.strip())
    if local_path.strip() and not _local_path_valid:
        st.warning("File not found at that path.")

    btn_col, _ = st.columns([2, 5])
    with btn_col:
        st.button(
            "▶  Run Pipeline",
            key="run_pipeline_btn",
            type="primary",
            disabled=(uploaded is None and not _local_path_valid),
            use_container_width=True,
        )

    if st.session_state.pipeline_result is not None:
        meta    = st.session_state.pipeline_result.video_metadata
        indexed = len(st.session_state.pipeline_result.frame_index_entries)
        mins    = int(meta.duration_sec // 60)
        secs    = int(meta.duration_sec % 60)
        _pt     = st.session_state.get("pipeline_time_sec")
        _from_cache = st.session_state.get("pipeline_from_cache", False)
        if _pt is not None:
            _pt_mins = int(_pt // 60)
            _pt_secs = int(_pt % 60)
            _pt_str  = f"{_pt_mins}m {_pt_secs}s" if _pt_mins else f"{_pt_secs}s"
            _pt_sub  = "from cache ⚡" if _from_cache else "wall clock"
        else:
            _pt_str, _pt_sub = "—", ""
        st.markdown(f"""
<div class="metric-row" style="margin-top:16px;">
  <div class="mc">
    <div class="mc-label">Duration</div>
    <div class="mc-value">{mins}:{secs:02d}</div>
    <div class="mc-sub">{meta.duration_sec:.0f} sec</div>
  </div>
  <div class="mc">
    <div class="mc-label">FPS</div>
    <div class="mc-value">{meta.fps:.0f}</div>
    <div class="mc-sub">frames/sec</div>
  </div>
  <div class="mc">
    <div class="mc-label">Resolution</div>
    <div class="mc-value" style="font-size:18px;">{meta.width}×{meta.height}</div>
    <div class="mc-sub">pixels</div>
  </div>
  <div class="mc">
    <div class="mc-label">Indexed</div>
    <div class="mc-value">{indexed}</div>
    <div class="mc-sub">of {meta.total_frames} frames</div>
  </div>
  <div class="mc">
    <div class="mc-label">Pipeline time</div>
    <div class="mc-value" style="font-size:20px;">{_pt_str}</div>
    <div class="mc-sub">{_pt_sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)

def _hash_file_chunked(path: str, chunk_mb: int = 8) -> str:
    """SHA256 of a file read in chunks — constant memory even for GB-sized videos."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_mb * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Pipeline execution (outside the container so rerun doesn't re-enter the border block mid-run)
if st.session_state.get("run_pipeline_btn") and (uploaded is not None or _local_path_valid):
    if uploaded is not None:
        # Read file bytes once — used for both content hash and temp file
        _file_bytes = uploaded.read()
        _content_hash = hashlib.sha256(_file_bytes).hexdigest()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as tmp:
            tmp.write(_file_bytes)
            tmp_path = tmp.name
        del _file_bytes  # free memory
    else:
        # Local path: no copy needed; hash in chunks to keep memory flat
        tmp_path = local_path.strip()
        _content_hash = _hash_file_chunked(tmp_path)

    st.session_state.video_path      = tmp_path
    st.session_state.pipeline_result = None
    st.session_state.pipeline_logs   = []
    st.session_state.pipeline_time_sec = None
    st.session_state.pipeline_from_cache = False

    from utils.config_loader import clear_cache as _clear_cfg_cache
    _clear_cfg_cache()
    config = get_config()
    config.anomaly.enable_rule_based = enable_rule_based

    log_records: list = st.session_state.pipeline_logs
    handler     = _ListHandler(log_records)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    old_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    _pipeline_ok = False
    _t_pipeline_start = time.time()
    try:
        _progress_bar = st.progress(0.0, text="Starting pipeline — YOLO · DeepSORT · CLIP …")

        def _on_progress(frac: float, msg: str) -> None:
            _progress_bar.progress(min(frac, 1.0), text=msg)

        pipeline = VideoPipeline(
            config,
            detector=_cached_yolo(),
            tracker=_cached_tracker(),
            encoder=_cached_clip(),
        )
        result: PipelineResult = pipeline.run(
            tmp_path, content_hash=_content_hash, progress_callback=_on_progress
        )
        _progress_bar.progress(1.0, text="Pipeline complete")
        st.session_state.pipeline_result = result
        st.session_state.pipeline_time_sec = time.time() - _t_pipeline_start
        # Detect if result came from cache (much faster than a full run)
        st.session_state.pipeline_from_cache = (
            st.session_state.pipeline_time_sec < 30.0
        )
        _pipeline_ok = True
    except Exception as exc:
        tb = traceback.format_exc()
        log_records.append(f"ERROR — {exc}")
        log_records.append(tb)
        st.error(f"Pipeline error: {exc}")
        with st.expander("Full traceback", expanded=True):
            st.code(tb, language="python")
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)

    if _pipeline_ok:
        meta = st.session_state.pipeline_result.video_metadata
        _cache_note = " (loaded from cache ⚡)" if st.session_state.pipeline_from_cache else ""
        st.success(
            f"Pipeline complete{_cache_note} · {meta.total_frames} frames · "
            f"{meta.duration_sec:.1f}s · {meta.fps:.1f} fps · "
            f"{len(st.session_state.pipeline_result.frame_index_entries)} frames indexed"
        )
        st.rerun()

if show_logs and st.session_state.pipeline_logs:
    with st.expander(
        f"Pipeline logs ({len(st.session_state.pipeline_logs)} lines)", expanded=False
    ):
        st.code("\n".join(st.session_state.pipeline_logs), language="text")


# ------------------------------------------------------------------ #
#  STEP 2 — Natural language query
# ------------------------------------------------------------------ #

with st.container(border=True):
    st.markdown('<div class="step-label">STEP 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Natural Language Query</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Query",
        key="query_text",
        placeholder='"person running near exit"  ·  "individual loitering by entrance"',
        label_visibility="collapsed",
        disabled=st.session_state.pipeline_result is None,
    )

    st.markdown(
        '<div style="font-size:12px;color:#94a3b8;margin:6px 0 4px;">Try a suggestion:</div>',
        unsafe_allow_html=True,
    )
    def _use_suggestion(text: str) -> None:
        # Runs as a button callback, i.e. before the query_text widget is
        # re-instantiated on the next run, so assigning to the widget's
        # session_state key here is permitted by Streamlit.
        st.session_state.query_text = text

    pill_cols = st.columns(len(QUERY_SUGGESTIONS))
    for col, suggestion in zip(pill_cols, QUERY_SUGGESTIONS):
        col.button(
            suggestion,
            key=f"pill_{suggestion}",
            use_container_width=True,
            disabled=st.session_state.pipeline_result is None,
            on_click=_use_suggestion,
            args=(suggestion,),
        )

    st.button(
        "Search",
        key="search_btn",
        type="primary",
        disabled=(st.session_state.pipeline_result is None or not query),
    )

if st.session_state.get("search_btn") and query and st.session_state.pipeline_result is not None:
    search_log_records: list = []
    _sh = _ListHandler(search_log_records)
    _root = logging.getLogger()
    _root.addHandler(_sh)
    _old = _root.level
    _root.setLevel(logging.DEBUG)

    try:
        with st.spinner("Encoding query · searching FAISS index …"):
            _app_logger.info("Search started — query: '%s'", query)
            qenc: QueryEncoder = _cached_query_encoder()
            _app_logger.info("Encoding text query …")
            q_vec = qenc.encode(query)
            _app_logger.info("Query encoded — shape %s", q_vec.shape)
            _pr: PipelineResult = st.session_state.pipeline_result
            _app_logger.info("Building FAISS index (%d vectors) …", len(_pr.frame_index_entries))
            _searcher = SimilaritySearch(get_config())
            _searcher.build_index(_pr.embedding_matrix, _pr.frame_index_entries)
            _app_logger.info("Searching FAISS index (top_k=%d) …", top_k)
            results = _searcher.search(q_vec, top_k=top_k)
            _app_logger.info("Search returned %d results", len(results))
            segments = localize_segments(
                results,
                gap_threshold_sec=gap_threshold,
                min_segment_duration_sec=min_seg_dur,
                min_score=min_score,
            )
            _app_logger.info("Localised %d segments", len(segments))
            st.session_state.search_results = results
            st.session_state.video_segments = segments

            # Full similarity timeline: query vs every indexed frame — one matmul.
            _sims = _pr.embedding_matrix @ q_vec
            st.session_state.timeline = {
                "timestamps": [e.timestamp_sec for e in _pr.frame_index_entries],
                "scores": _sims.tolist(),
                "query": query,
                "min_score": min_score,
            }
    except Exception as exc:
        tb = traceback.format_exc()
        _app_logger.error("Search failed: %s", exc)
        st.error(f"Search error: {exc}")
        with st.expander("Search traceback", expanded=True):
            st.code(tb, language="python")
    finally:
        _root.removeHandler(_sh)
        _root.setLevel(_old)

    if search_log_records:
        with st.expander(f"Search logs ({len(search_log_records)} lines)", expanded=False):
            st.code("\n".join(search_log_records), language="text")


# ------------------------------------------------------------------ #
#  STEP 3 — Matching segments
# ------------------------------------------------------------------ #

if st.session_state.video_segments is not None:
    segments: List[VideoSegment] = st.session_state.video_segments

    with st.container(border=True):
        count_str = f'<span style="color:#3b82f6;">{len(segments)} found</span>'
        st.markdown('<div class="step-label">STEP 3</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="step-title">Matching Segments ({count_str})</div>',
            unsafe_allow_html=True,
        )

        # Similarity timeline: where in the whole video does the query match?
        _tl = st.session_state.get("timeline")
        if _tl and _tl.get("scores"):
            try:
                import altair as alt
                import pandas as pd

                _tl_df = pd.DataFrame({
                    "time_min": [t / 60.0 for t in _tl["timestamps"]],
                    "similarity": _tl["scores"],
                })
                base = alt.Chart(_tl_df).mark_area(
                    line={"color": "#3b82f6"},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[alt.GradientStop(color="#dbeafe", offset=0),
                               alt.GradientStop(color="#93c5fd", offset=1)],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                ).encode(
                    x=alt.X("time_min:Q", title="Video time (minutes)"),
                    y=alt.Y("similarity:Q", title="Cosine similarity",
                            scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("time_min:Q", format=".2f", title="min"),
                             alt.Tooltip("similarity:Q", format=".3f")],
                )
                threshold = alt.Chart(
                    pd.DataFrame({"y": [_tl.get("min_score", 0.2)]})
                ).mark_rule(color="#ef4444", strokeDash=[5, 4]).encode(y="y:Q")

                layers = [base, threshold]
                if segments:
                    _seg_df = pd.DataFrame({
                        "start": [s.start_sec / 60.0 for s in segments[:10]],
                        "end": [max(s.end_sec, s.start_sec + 1) / 60.0 for s in segments[:10]],
                    })
                    layers.append(
                        alt.Chart(_seg_df).mark_rect(
                            color="#22c55e", opacity=0.18
                        ).encode(x="start:Q", x2="end:Q")
                    )
                _anoms = st.session_state.pipeline_result.anomaly_events
                if _anoms:
                    _an_df = pd.DataFrame({"t": [a.start_sec / 60.0 for a in _anoms]})
                    layers.append(
                        alt.Chart(_an_df).mark_rule(
                            color="#f59e0b", strokeWidth=2
                        ).encode(x="t:Q")
                    )
                st.altair_chart(
                    alt.layer(*layers).properties(height=180),
                    use_container_width=True,
                )
                st.caption(
                    f'Similarity of "{_tl["query"]}" across the full video · '
                    "green bands = matched segments · red dashes = score threshold"
                    + (" · orange lines = anomaly events" if _anoms else "")
                )
            except Exception as _tl_exc:
                st.caption(f"Timeline unavailable: {_tl_exc}")

        if not segments:
            st.info("No segments matched. Try adjusting search params or rephrasing the query.")

        for i, seg in enumerate(segments[:10]):
            bar_pct = min(int(seg.peak_score * 100), 100)

            with st.expander(
                f"Segment {i + 1}  ·  {seg.start_sec:.1f}s – {seg.end_sec:.1f}s"
                f"  ·  {seg.duration_sec:.1f}s  ·  score {seg.peak_score:.3f}",
                expanded=(i == 0),
            ):
                st.markdown(f"""
<div class="seg-header">
  <span class="seg-title">Segment {i + 1} of {len(segments)}</span>
  <span class="seg-badge">Score: {seg.peak_score:.3f}</span>
</div>
<div class="seg-meta">
  ⏱ {seg.start_sec:.1f}s – {seg.end_sec:.1f}s &nbsp;·&nbsp;
  Duration: {seg.duration_sec:.1f}s &nbsp;·&nbsp;
  {len(seg.frame_indices)} frames matched
</div>
<div class="score-bar">
  <div class="score-fill" style="width:{bar_pct}%;"></div>
</div>
""", unsafe_allow_html=True)

                if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                    try:
                        frames_to_show   = min(4, len(seg.frame_indices))
                        step             = max(1, len(seg.frame_indices) // frames_to_show)
                        selected_indices = seg.frame_indices[::step][:frames_to_show]

                        img_cols  = st.columns(len(selected_indices))
                        cap       = cv2.VideoCapture(st.session_state.video_path)
                        pr: PipelineResult = st.session_state.pipeline_result
                        _q_text   = st.session_state.get("query_text", "").lower()
                        _highlight = _query_highlight_classes(_q_text)

                        # Determine track-level filter mode
                        _is_collision = any(kw in _q_text for kw in _COLLISION_KEYWORDS)
                        _color_target = next(
                            (c for c in _COLOR_MASKS_BGR if c in _q_text), None
                        )

                        # Query-aware open-vocab boxes: top segment only (CPU cost)
                        _ov_detector = None
                        _ov_vocab: list = []
                        if i == 0 and _q_text and get_config().retrieval.query_aware_detection:
                            from models.open_vocab_detector import query_to_vocabulary
                            _ov_vocab = query_to_vocabulary(_q_text)
                            if _ov_vocab:
                                _ov_detector = _cached_open_vocab()

                        for col, frame_idx in zip(img_cols, selected_indices):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame_bgr = cap.read()
                            if ret:
                                try:
                                    if _is_collision:
                                        _tids = _collision_track_ids(frame_idx, pr)
                                    elif _color_target:
                                        _tids = _color_car_track_ids(
                                            frame_bgr, frame_idx, pr, _color_target
                                        )
                                    else:
                                        _tids = None
                                    _draw_tracks(frame_bgr, frame_idx, pr,
                                                 highlight_classes=_highlight,
                                                 highlight_track_ids=_tids)
                                    if _ov_detector is not None:
                                        _draw_open_vocab(
                                            frame_bgr,
                                            _ov_detector.detect(frame_bgr, _ov_vocab),
                                        )
                                except Exception:
                                    pass
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                col.image(
                                    Image.fromarray(frame_rgb),
                                    caption=f"t = {frame_idx / pr.video_metadata.fps:.1f}s",
                                    use_column_width=True,
                                )
                        cap.release()
                        if _ov_detector is not None:
                            st.caption(
                                "🟣 Magenta boxes: YOLO-World open-vocabulary detections "
                                f"for your query terms ({', '.join(_ov_vocab[1:] or _ov_vocab)})"
                            )

                        # Click-to-play: embedded player that jumps to this segment
                        if st.checkbox(
                            f"▶ Play this segment ({seg.start_sec:.0f}s – {seg.end_sec:.0f}s)",
                            key=f"play_seg_{i}",
                        ):
                            try:
                                st.video(
                                    st.session_state.video_path,
                                    start_time=int(seg.start_sec),
                                    end_time=int(seg.end_sec) + 2,
                                    muted=True,
                                    autoplay=True,
                                )
                            except Exception as _vid_exc:
                                st.caption(
                                    f"In-browser playback unavailable ({_vid_exc}). "
                                    "Browser playback requires an H.264 .mp4 file."
                                )
                    except Exception as _seg_exc:
                        st.caption(f"Frame preview unavailable: {_seg_exc}")


# ------------------------------------------------------------------ #
#  STEP 4 — Anomaly events
# ------------------------------------------------------------------ #

if st.session_state.pipeline_result is not None:
    try:
        _anom_result: PipelineResult = st.session_state.pipeline_result
        events: List[AnomalyEvent] = _anom_result.anomaly_events

        with st.container(border=True):
            n     = len(events)
            color = "#ef4444" if n > 0 else "#22c55e"
            st.markdown('<div class="step-label">STEP 4</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="step-title">Anomaly Events '
                f'(<span style="color:{color};">{n} detected</span>)</div>',
                unsafe_allow_html=True,
            )

            if not events:
                st.success("No anomaly events detected in this video.")
            else:
                _video_path = st.session_state.get("video_path")
                _fps = _anom_result.video_metadata.fps or 25.0

                # Open video once for all thumbnails
                _anom_cap = None
                if _video_path and os.path.exists(_video_path):
                    try:
                        _anom_cap = cv2.VideoCapture(_video_path)
                    except Exception:
                        _anom_cap = None

                for e in events:
                    sev = e.severity
                    if sev >= 0.7:
                        sev_cls, sev_label, icon = "sev-H", "HIGH",   "🔴"
                    elif sev >= 0.4:
                        sev_cls, sev_label, icon = "sev-M", "MEDIUM", "🟡"
                    else:
                        sev_cls, sev_label, icon = "sev-L", "LOW",    "🟢"

                    duration = e.end_sec - e.start_sec

                    thumb_col, info_col = st.columns([1, 3])

                    # Frame thumbnail at event start
                    with thumb_col:
                        if _anom_cap is not None:
                            try:
                                _target_frame = int(e.start_sec * _fps)
                                _anom_cap.set(cv2.CAP_PROP_POS_FRAMES, _target_frame)
                                _ret, _frame_bgr = _anom_cap.read()
                                if _ret:
                                    # Draw closest bbox for this track
                                    _track_snaps = _anom_result.track_histories.get(e.track_id, [])
                                    _best_snap = None
                                    _best_dt = float("inf")
                                    for _snap in _track_snaps:
                                        _dt = abs(_snap.timestamp_sec - e.start_sec)
                                        if _dt < _best_dt:
                                            _best_dt = _dt
                                            _best_snap = _snap
                                    if _best_snap is not None and _best_dt < 2.0:
                                        _bx1, _by1, _bx2, _by2 = [int(v) for v in _best_snap.bbox_xyxy]
                                        cv2.rectangle(_frame_bgr, (_bx1, _by1), (_bx2, _by2), (239, 68, 68), 2)
                                        cv2.putText(
                                            _frame_bgr, f"ID:{e.track_id}",
                                            (_bx1, max(_by1 - 6, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239, 68, 68), 1,
                                        )
                                    _frame_rgb = cv2.cvtColor(_frame_bgr, cv2.COLOR_BGR2RGB)
                                    st.image(Image.fromarray(_frame_rgb), use_column_width=True)
                            except Exception as _thumb_exc:
                                st.caption(f"No preview: {_thumb_exc}")

                if _anom_cap is not None:
                    _anom_cap.release()

                    # Event info card
                    with info_col:
                        st.markdown(f"""
<div class="anom-card" style="margin-bottom:0;height:100%;">
  <div style="font-size:28px;min-width:36px;text-align:center;">{icon}</div>
  <div class="anom-info">
    <div class="anom-type">{e.event_type.upper()}</div>
    <div class="anom-meta">
      Track #{e.track_id} &nbsp;·&nbsp;
      {e.start_sec:.1f}s – {e.end_sec:.1f}s &nbsp;·&nbsp;
      {duration:.1f}s &nbsp;·&nbsp;
      Location ({e.location_xy[0]:.0f}, {e.location_xy[1]:.0f})
    </div>
  </div>
  <span class="{sev_cls}">{sev_label}</span>
</div>
""", unsafe_allow_html=True)

    except Exception as _anom_exc:
        with st.container(border=True):
            st.markdown('<div class="step-label">STEP 4</div>', unsafe_allow_html=True)
            st.error(f"Anomaly section error: {_anom_exc}")
            with st.expander("Traceback", expanded=False):
                st.code(traceback.format_exc(), language="python")
