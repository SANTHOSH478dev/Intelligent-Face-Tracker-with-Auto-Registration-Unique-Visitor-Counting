"""InsightFace (ArcFace) embedding generator + in-memory gallery matcher."""
from typing import List, Optional, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class FaceRecognizer:
    def __init__(self, model_name: str, device: str):
        ctx = -1 if device == "cpu" else int(device)
        providers = (["CPUExecutionProvider"] if ctx < 0
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        app = FaceAnalysis(name=model_name, providers=providers,
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=ctx, det_size=(320, 320))
        self.rec = app.models["recognition"]   # ArcFace R50 (512-d)

    def embed(self, frame, box, kps) -> np.ndarray:
        """Aligned (5-pt) 112x112 crop -> L2-normalised 512-d embedding."""
        if kps is not None:
            aligned = face_align.norm_crop(frame, kps.astype(np.float32), 112)
        else:  # fallback: padded crop
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box.astype(int)
            pw, ph = int((x2 - x1) * .15), int((y2 - y1) * .15)
            crop = frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]
            aligned = cv2.resize(crop, (112, 112))
        feat = self.rec.get_feat(aligned).flatten()
        return feat / (np.linalg.norm(feat) + 1e-9)


class Gallery:
    """Known identities; cosine-similarity search."""

    def __init__(self, ids: List[int], embs: List[np.ndarray], threshold: float):
        self.ids, self.embs, self.thr = list(ids), list(embs), threshold

    def match(self, emb) -> Tuple[Optional[int], float]:
        if not self.ids:
            return None, 0.0
        sims = np.stack(self.embs) @ emb
        i = int(np.argmax(sims))
        return (self.ids[i], float(sims[i])) if sims[i] >= self.thr else (None, float(sims[i]))

    def add(self, face_id: int, emb: np.ndarray):
        self.ids.append(face_id)
        self.embs.append(emb)
