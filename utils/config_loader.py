"""Loads config.yaml into a typed AppConfig dataclass. Singleton-cached."""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml

_config_cache = {}


@dataclass
class PipelineConfig:
    frame_skip: int = 5
    batch_size: int = 32
    cache_dir: str = ".cache"


@dataclass
class VideoConfig:
    max_resolution: List[int] = field(default_factory=lambda: [1280, 720])


@dataclass
class YOLOConfig:
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    class_whitelist: List[int] = field(default_factory=lambda: [0, 2, 3, 5, 7])
    device: str = "auto"


@dataclass
class DeepSORTConfig:
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.3
    nn_budget: int = 100


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "auto"
    query_prefix: str = "a photo of"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    gap_threshold_sec: float = 2.0
    min_segment_duration_sec: float = 1.0


@dataclass
class LoiteringConfig:
    dwell_radius_px: int = 80
    dwell_time_sec: float = 10.0


@dataclass
class IntrusionConfig:
    roi_zones: List[List[List[int]]] = field(default_factory=list)


@dataclass
class AnomalyConfig:
    enable_rule_based: bool = True
    enable_vadclip: bool = False
    vadclip_weights: str = "weights/vadclip_ucf.pth"
    loitering: LoiteringConfig = field(default_factory=LoiteringConfig)
    intrusion: IntrusionConfig = field(default_factory=IntrusionConfig)


@dataclass
class EvaluationConfig:
    iou_threshold: float = 0.5
    dataset_root: str = "data/datasets"
    ucf_crime_root: str = "data/datasets/UCF_Crimes"
    mot17_root: str = "data/datasets/MOT17"


@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    deepsort: DeepSORTConfig = field(default_factory=DeepSORTConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _dict_to_config(d: dict, cls):
    """Recursively map a nested dict to a dataclass, ignoring unknown keys."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return d
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        if dataclasses.is_dataclass(f.type) or (isinstance(f.type, type) and dataclasses.is_dataclass(f.type)):
            kwargs[f.name] = _dict_to_config(val, f.type)
        else:
            # Resolve string annotations
            try:
                import typing
                origin = getattr(f.type, "__origin__", None)
            except Exception:
                origin = None
            kwargs[f.name] = val
    return cls(**kwargs)


def get_config(path: str = "configs/config.yaml") -> AppConfig:
    """Return the AppConfig singleton for the given path."""
    if path in _config_cache:
        return _config_cache[path]

    if not os.path.exists(path):
        cfg = AppConfig()
        _config_cache[path] = cfg
        return cfg

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def _build(raw_section, cls):
        import dataclasses
        if not dataclasses.is_dataclass(cls):
            return raw_section
        kwargs = {}
        for fld in dataclasses.fields(cls):
            if fld.name not in raw_section:
                continue
            val = raw_section[fld.name]
            field_type = fld.type
            # Resolve string annotations
            if isinstance(field_type, str):
                field_type = eval(field_type)
            if dataclasses.is_dataclass(field_type) and isinstance(val, dict):
                kwargs[fld.name] = _build(val, field_type)
            else:
                kwargs[fld.name] = val
        return cls(**kwargs)

    cfg = _build(raw, AppConfig)
    _config_cache[path] = cfg
    return cfg


def clear_cache():
    _config_cache.clear()
