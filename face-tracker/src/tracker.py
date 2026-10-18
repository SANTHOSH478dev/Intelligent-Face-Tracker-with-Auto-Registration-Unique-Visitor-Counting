"""Track bookkeeping on top of ByteTrack IDs: hits, missed cycles, lost detection."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(eq=False)
class Track:
    track_id: int
    box: np.ndarray
    kps: Optional[np.ndarray]
    face_id: Optional[int] = None      # set once recognised/registered
    hits: int = 1
    missed: int = 0
    silent: bool = False               # duplicate of an already-active face: no entry/exit
    last_crop: Optional[np.ndarray] = None


class FaceTracker:
    def __init__(self, max_missed: int):
        self.max_missed = max_missed
        self.tracks: List[Track] = []

    def update(self, dets):
        """dets carry ByteTrack IDs. Returns (matched[(track, det)], new_tracks, lost_tracks)."""
        by_id = {t.track_id: t for t in self.tracks}
        seen, matched, new = set(), [], []
        for d in dets:
            t = by_id.get(d.track_id)
            if t is None:
                t = Track(d.track_id, d.box, d.kps)
                self.tracks.append(t); new.append(t)
            else:
                t.box, t.kps, t.hits, t.missed = d.box, d.kps, t.hits + 1, 0
                matched.append((t, d))
            seen.add(d.track_id)

        lost = []
        for t in list(self.tracks):
            if t.track_id in seen:
                continue
            t.missed += 1
            if t.missed > self.max_missed:
                lost.append(t); self.tracks.remove(t)
        return matched, new, lost
