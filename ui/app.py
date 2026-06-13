"""Streamlit demo app for the Smart Query-Driven Surveillance System."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
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
#  Page config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Smart Surveillance — Query-Driven Video Retrieval",
    page_icon="🎥",
    layout="wide",
)

st.title("Smart Query-Driven Surveillance System")
st.caption(
    "Vision-Language Based Video Retrieval + Suspicious Activity Detection | "
    "M.Tech Dissertation BITS WILP 2026"
)


# ------------------------------------------------------------------ #
#  Log capture — attaches to root logger during pipeline run
# ------------------------------------------------------------------ #

class _ListHandler(logging.Handler):
    """Appends formatted log records to a list."""
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
        "faiss_index": None,
        "video_path": None,
        "query_encoder": None,
        "pipeline_logs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _draw_tracks(frame_bgr: np.ndarray, frame_idx: int, result: PipelineResult):
    for track_id, snapshots in result.track_histories.items():
        for snap in snapshots:
            if snap.frame_index == frame_idx:
                x1, y1, x2, y2 = [int(v) for v in snap.bbox_xyxy]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr, f"ID:{track_id}",
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Top-K results", 1, 30, 10)
    gap_threshold = st.slider("Segment gap threshold (sec)", 0.5, 10.0, 2.0, 0.5)
    min_seg_dur = st.slider("Min segment duration (sec)", 0.0, 5.0, 1.0, 0.5)
    st.markdown("---")
    st.markdown("**Anomaly Detection**")
    enable_rule_based = st.checkbox("Rule-based (loitering / intrusion)", value=True)
    st.caption("VadCLIP requires pretrained weights — see CLAUDE.md.")
    st.markdown("---")
    show_logs = st.checkbox("Show pipeline logs", value=True)


# ------------------------------------------------------------------ #
#  Upload + pipeline
# ------------------------------------------------------------------ #

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("1. Upload Surveillance Video")
    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mkv", "mov", "mpeg4"],
    )
    run_pipeline = st.button("Run Pipeline", type="primary", disabled=uploaded is None)

    if run_pipeline and uploaded is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded.name)[1]
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.session_state.video_path = tmp_path
        st.session_state.pipeline_result = None
        st.session_state.faiss_index = None
        st.session_state.pipeline_logs = []

        config = get_config()
        config.anomaly.enable_rule_based = enable_rule_based

        log_records: list = st.session_state.pipeline_logs
        handler = _ListHandler(log_records)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        old_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        try:
            with st.spinner("Processing video (YOLO + DeepSORT + CLIP)..."):
                pipeline = VideoPipeline(config)
                result: PipelineResult = pipeline.run(tmp_path)
                st.session_state.pipeline_result = result

                search = SimilaritySearch(config)
                search.build_index(result.embedding_matrix, result.frame_index_entries)
                st.session_state.faiss_index = search

                st.session_state.query_encoder = QueryEncoder(config=config)

                meta = result.video_metadata
                st.success(
                    f"Pipeline complete! {meta.total_frames} frames, "
                    f"{meta.duration_sec:.1f}s, {meta.fps:.1f} fps — "
                    f"{len(result.frame_index_entries)} frames indexed."
                )
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

with col2:
    st.subheader("Video Metadata")
    if st.session_state.pipeline_result is not None:
        meta = st.session_state.pipeline_result.video_metadata
        st.json({
            "Duration (sec)": round(meta.duration_sec, 2),
            "FPS": round(meta.fps, 2),
            "Resolution": f"{meta.width}x{meta.height}",
            "Total frames": meta.total_frames,
            "Frames indexed": len(st.session_state.pipeline_result.frame_index_entries),
        })
    else:
        st.info("Upload a video and click 'Run Pipeline' to start.")


# ------------------------------------------------------------------ #
#  Pipeline logs panel
# ------------------------------------------------------------------ #

if show_logs and st.session_state.pipeline_logs:
    with st.expander(
        f"Pipeline logs ({len(st.session_state.pipeline_logs)} lines)", expanded=False
    ):
        st.code("\n".join(st.session_state.pipeline_logs), language="text")


# ------------------------------------------------------------------ #
#  Query + retrieval
# ------------------------------------------------------------------ #

st.markdown("---")
st.subheader("2. Natural Language Query")

query_col, btn_col = st.columns([5, 1])
with query_col:
    query = st.text_input(
        "Enter your query",
        placeholder='"person running near exit" or "individual loitering by entrance"',
        label_visibility="collapsed",
    )
with btn_col:
    search_btn = st.button(
        "Search",
        type="primary",
        disabled=(st.session_state.faiss_index is None or not query),
    )

if search_btn and query and st.session_state.faiss_index is not None:
    with st.spinner("Encoding query and searching..."):
        qenc: QueryEncoder = st.session_state.query_encoder
        q_vec = qenc.encode(query)
        search: SimilaritySearch = st.session_state.faiss_index
        results = search.search(q_vec, top_k=top_k)
        segments = localize_segments(
            results,
            gap_threshold_sec=gap_threshold,
            min_segment_duration_sec=min_seg_dur,
        )
        st.session_state.search_results = results
        st.session_state.video_segments = segments


# ------------------------------------------------------------------ #
#  Results — matching segments
# ------------------------------------------------------------------ #

if st.session_state.video_segments:
    segments: List[VideoSegment] = st.session_state.video_segments
    st.subheader(f"3. Matching Segments ({len(segments)} found)")

    for i, seg in enumerate(segments[:10]):
        with st.expander(
            f"Segment {i+1} | {seg.start_sec:.1f}s – {seg.end_sec:.1f}s "
            f"| score: {seg.peak_score:.3f} | duration: {seg.duration_sec:.1f}s",
            expanded=(i == 0),
        ):
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                frames_to_show = min(4, len(seg.frame_indices))
                step = max(1, len(seg.frame_indices) // frames_to_show)
                selected_indices = seg.frame_indices[::step][:frames_to_show]

                cols = st.columns(len(selected_indices))
                cap = cv2.VideoCapture(st.session_state.video_path)
                result: PipelineResult = st.session_state.pipeline_result

                for col, frame_idx in zip(cols, selected_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame_bgr = cap.read()
                    if ret:
                        _draw_tracks(frame_bgr, frame_idx, result)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        col.image(
                            Image.fromarray(frame_rgb),
                            caption=f"t={frame_idx / result.video_metadata.fps:.1f}s",
                            use_column_width=True,
                        )
                cap.release()


# ------------------------------------------------------------------ #
#  Anomaly events
# ------------------------------------------------------------------ #

if st.session_state.pipeline_result is not None:
    result: PipelineResult = st.session_state.pipeline_result
    events: List[AnomalyEvent] = result.anomaly_events

    st.markdown("---")
    st.subheader("4. Anomaly Events")

    if not events:
        st.info("No anomaly events detected.")
    else:
        import pandas as pd
        rows = [
            {
                "Type": e.event_type,
                "Track ID": e.track_id,
                "Start (sec)": round(e.start_sec, 2),
                "End (sec)": round(e.end_sec, 2),
                "Duration (sec)": round(e.end_sec - e.start_sec, 2),
                "Severity": round(e.severity, 2),
                "Location (x, y)": f"({e.location_xy[0]:.0f}, {e.location_xy[1]:.0f})",
            }
            for e in events
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(f"{len(events)} events detected.")
