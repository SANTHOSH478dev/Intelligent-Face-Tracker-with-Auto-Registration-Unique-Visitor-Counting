"""SQLite persistence. Every write commits immediately (WAL) so a crash loses nothing."""
import os
import sqlite3
import numpy as np


class Database:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS faces(
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT);
        CREATE TABLE IF NOT EXISTS events(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
            timestamp TEXT NOT NULL,
            frame_idx INTEGER,
            image_path TEXT,
            FOREIGN KEY(face_id) REFERENCES faces(face_id));
        CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
        """)
        self.conn.commit()

    def register_face(self, ts: str, emb: np.ndarray, image_path: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO faces(first_seen, embedding, image_path) VALUES (?,?,?)",
            (ts, emb.astype(np.float32).tobytes(), image_path))
        self.conn.commit()
        return cur.lastrowid

    def set_face_image(self, face_id: int, path: str):
        self.conn.execute("UPDATE faces SET image_path=? WHERE face_id=?", (path, face_id))
        self.conn.commit()

    def log_event(self, face_id, etype, ts, frame_idx, image_path):
        self.conn.execute(
            "INSERT INTO events(face_id,event_type,timestamp,frame_idx,image_path) "
            "VALUES (?,?,?,?,?)", (face_id, etype, ts, frame_idx, image_path))
        self.conn.commit()

    def load_gallery(self):
        rows = self.conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        return [r[0] for r in rows], [np.frombuffer(r[1], dtype=np.float32) for r in rows]

    def unique_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def event_counts(self):
        return dict(self.conn.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall())

    def close(self):
        self.conn.close()
