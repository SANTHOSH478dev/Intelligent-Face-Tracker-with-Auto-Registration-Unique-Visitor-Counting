"""Loads and validates config.json."""
import json
from dataclasses import dataclass, fields


@dataclass
class Config:
    source: str = "data/sample.mp4"       # video path or rtsp:// URL
    is_stream: bool = False               # True for RTSP -> auto-reconnect
    frame_skip: int = 2                   # frames skipped between detection cycles
    yolo_model: str = "models/yolov8n-face.pt"
    detection_conf: float = 0.5
    min_face_size: int = 40               # px; ignore tiny faces
    min_sharpness: float = 25.0           # Laplacian variance; prefer sharp crops for embedding
    blur_fallback_hits: int = 8           # blurry face seen this many times => register anyway
    device: str = "cpu"                   # "cpu" or "0" (GPU index)
    recognition_model: str = "buffalo_l"  # InsightFace pack (ArcFace r50)
    recognition_threshold: float = 0.40   # cosine similarity => same identity
    max_missed_cycles: int = 10           # detection cycles w/o match => track lost
    min_confirm_hits: int = 3             # hits before embedding/registration
    exit_grace_frames: int = 60           # re-appearing within this => same visit
    reconnect_delay_sec: int = 5
    log_dir: str = "logs"
    db_path: str = "data/visitors.db"
    display: bool = False


def load_config(path: str = "config.json") -> Config:
    with open(path) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(Config)}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return Config(**raw)
