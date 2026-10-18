"""Orchestrates detect+track -> recognise/register -> entry/exit logging."""
import os
import logging
from datetime import datetime
import cv2
import numpy as np

from .database import Database
from .recognizer import Gallery
from .tracker import FaceTracker, Track

log = logging.getLogger("facetracker")
NAMES = {"entries": "entry", "exits": "exit", "faces": "face"}


def _crop(frame, box, pad=0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return frame[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)].copy()


class FaceCounterPipeline:
    def __init__(self, cfg, db: Database, detector=None, recognizer=None):
        self.cfg, self.db = cfg, db
        if detector is None:                       # lazy heavy imports (injectable for tests)
            from .detector import FaceDetector
            detector = FaceDetector(cfg.yolo_model, cfg.detection_conf,
                                    cfg.min_face_size, cfg.device)
        if recognizer is None:
            from .recognizer import FaceRecognizer
            recognizer = FaceRecognizer(cfg.recognition_model, cfg.device)
        self.detector, self.recognizer = detector, recognizer
        ids, embs = db.load_gallery()              # resume known identities after restart
        self.gallery = Gallery(ids, embs, cfg.recognition_threshold)
        self.tracker = FaceTracker(cfg.max_missed_cycles)
        self.frame_idx = 0
        self.pending = {}   # face_id -> (track, lost_time, lost_frame): exit held for grace period
        log.info("Pipeline ready | gallery=%d known faces", len(ids))

    # ---------- helpers ----------
    def _sharp(self, crop) -> bool:
        """Skip blurry crops: bad embeddings are the main cause of duplicate IDs."""
        if crop is None or crop.size == 0:
            return False
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
        return cv2.Laplacian(g, cv2.CV_64F).var() >= self.cfg.min_sharpness

    def _save(self, kind: str, crop, face_id: int, ts: datetime) -> str:
        d = os.path.join(self.cfg.log_dir, kind, ts.strftime("%Y-%m-%d"))
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{ts.strftime('%H-%M-%S-%f')[:-3]}_id{face_id}_{NAMES[kind]}.jpg")
        cv2.imwrite(path, crop)
        return path

    def _event(self, etype: str, track: Track, ts: datetime):
        kind = "entries" if etype == "entry" else "exits"
        crop = track.last_crop if track.last_crop is not None and track.last_crop.size \
            else np.zeros((112, 112, 3), np.uint8)
        path = self._save(kind, crop, track.face_id, ts)
        self.db.log_event(track.face_id, etype, ts.isoformat(timespec="milliseconds"),
                          self.frame_idx, path)
        log.info("%s | face_id=%d track=%d frame=%d img=%s",
                 etype.upper(), track.face_id, track.track_id, self.frame_idx, path)

    def _identify(self, frame, track: Track):
        ts = datetime.now()
        emb = self.recognizer.embed(frame, track.box, track.kps)
        log.info("EMBEDDING generated | track=%d dim=%d", track.track_id, emb.shape[0])
        fid, sim = self.gallery.match(emb)
        if fid is None:
            fid = self.db.register_face(ts.isoformat(timespec="milliseconds"), emb)
            self.db.set_face_image(fid, self._save("faces", track.last_crop, fid, ts))
            self.gallery.add(fid, emb)
            log.info("REGISTERED new face | face_id=%d best_sim=%.3f total_unique=%d",
                     fid, sim, self.db.unique_count())
        else:
            log.info("RECOGNIZED | face_id=%d sim=%.3f track=%d", fid, sim, track.track_id)
        track.face_id = fid
        if fid in self.pending:                    # came back within grace period
            self.pending.pop(fid)
            log.info("RESUMED | face_id=%d (same visit, no new entry/exit)", fid)
        elif any(o.face_id == fid for o in self.tracker.tracks if o is not track):
            track.silent = True                    # same person already tracked elsewhere
            log.info("MERGED | face_id=%d already tracked by another track", fid)
        else:
            self._event("entry", track, ts)

    # ---------- main step ----------
    def process(self, frame, frame_idx: int):
        self.frame_idx = frame_idx
        dets = self.detector.track(frame)
        matched, new, lost = self.tracker.update(dets)
        for t in [t for t, _ in matched] + new:
            t.last_crop = _crop(frame, t.box)
            # Prefer a sharp crop; if the face stays blurry, register it anyway so nobody is missed
            if t.face_id is None and t.hits >= self.cfg.min_confirm_hits and \
                    (self._sharp(t.last_crop) or t.hits >= self.cfg.blur_fallback_hits):
                self._identify(frame, t)
            elif t.face_id is not None and t.hits % 10 == 0:
                log.info("TRACKING | face_id=%d track=%d hits=%d frame=%d",
                         t.face_id, t.track_id, t.hits, frame_idx)
        for t in lost:                             # hold the exit during the grace period
            if t.face_id is not None and not t.silent:
                self.pending[t.face_id] = (t, datetime.now(), frame_idx)
        for fid, (t, ts, f0) in list(self.pending.items()):
            if frame_idx - f0 >= self.cfg.exit_grace_frames:
                self._event("exit", t, ts)
                del self.pending[fid]
        return self.tracker.tracks

    def close(self):
        """Flush all remaining exits at end of video / shutdown (exactly one exit per entry)."""
        for fid, (t, ts, _) in list(self.pending.items()):
            self._event("exit", t, ts)
        self.pending.clear()
        for t in list(self.tracker.tracks):
            if t.face_id is not None and not t.silent:
                self._event("exit", t, datetime.now())
        self.tracker.tracks.clear()
        log.info("FINAL unique visitors = %d | events=%s",
                 self.db.unique_count(), self.db.event_counts())
