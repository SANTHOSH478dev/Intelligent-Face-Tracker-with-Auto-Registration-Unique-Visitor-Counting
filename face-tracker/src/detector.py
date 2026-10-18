"""YOLOv8-face detector with built-in ByteTrack IDs (boxes, conf, landmarks, track_id)."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    box: np.ndarray                 # x1,y1,x2,y2
    conf: float
    kps: Optional[np.ndarray]       # (5,2) landmarks or None
    track_id: Optional[int] = None  # ByteTrack ID


class FaceDetector:
    def __init__(self, model_path: str, conf: float, min_size: int, device: str):
        self.model = YOLO(model_path)
        self.conf, self.min_size, self.device = conf, min_size, device

    def track(self, frame) -> List[Detection]:
        """Detect + ByteTrack association in one call (state kept via persist=True)."""
        res = self.model.track(frame, conf=self.conf, device=self.device,
                               persist=True, tracker="bytetrack.yaml",
                               verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.boxes.id is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy()
        kps = None
        if res.keypoints is not None and res.keypoints.xy is not None \
                and len(res.keypoints.xy) == len(boxes):
            kps = res.keypoints.xy.cpu().numpy()
        out = []
        for i, b in enumerate(boxes):
            if min(b[2] - b[0], b[3] - b[1]) < self.min_size:
                continue
            out.append(Detection(b, float(confs[i]),
                                 kps[i] if kps is not None else None, int(ids[i])))
        return out
