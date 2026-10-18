"""Simulated end-to-end test (no models/video needed):  python tests/test_logic.py"""
import os, sys, tempfile, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
for m in ("ultralytics", "insightface", "insightface.app", "insightface.utils"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["ultralytics"].YOLO = None
sys.modules["insightface.app"].FaceAnalysis = None
sys.modules["insightface.utils"].face_align = None
import numpy as np
from src.config import Config
from src.database import Database
from src.detector import Detection
from src.pipeline import FaceCounterPipeline

rng = np.random.default_rng(0)
FRAME = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)       # sharp noise image
VEC = {1: np.eye(8)[0], 2: np.eye(8)[1]}                          # person A, person B


class FakeDet:
    def __init__(self, schedule): self.s, self.i = schedule, 0
    def track(self, frame):
        ids = self.s[self.i]; self.i += 1
        return [Detection(np.array([20., 20, 120, 120]), .9, None, tid) for tid in ids]


class FakeRec:
    def __init__(self, det): self.det = det
    def embed(self, frame, box, kps): return VEC[self.det.last_pid]


tmp = tempfile.mkdtemp()
cfg = Config(log_dir=os.path.join(tmp, "logs"), db_path=os.path.join(tmp, "t.db"),
             max_missed_cycles=2, min_confirm_hits=2, exit_grace_frames=30)
# Person A (track 10) -> brief flicker (track 11, same person) -> B (track 12) -> A returns late (track 13)
sched = [[10]]*3 + [[]]*3 + [[11]]*3 + [[]]*8 + [[12]]*3 + [[]]*12 + [[13]]*3 + [[]]*8
det = FakeDet(sched)
pid_of = {10: 1, 11: 1, 12: 2, 13: 1}
rec = FakeRec(det)
db = Database(cfg.db_path)
pipe = FaceCounterPipeline(cfg, db, detector=det, recognizer=rec)
orig = det.track
def tracked(frame):
    out = orig(frame)
    rec.det.last_pid = pid_of[out[0].track_id] if out else 1
    return out
det.track = tracked
for i in range(len(sched)):
    pipe.process(FRAME, (i + 1) * 3)
pipe.close()
c = db.event_counts()
print("unique:", db.unique_count(), "events:", c)
assert db.unique_count() == 2, "A and B only"
assert c["entry"] == c["exit"], "one exit per entry"
assert c["entry"] == 3, "A (visit 1, flicker merged), B, A again after grace"
print("TEST PASSED")
