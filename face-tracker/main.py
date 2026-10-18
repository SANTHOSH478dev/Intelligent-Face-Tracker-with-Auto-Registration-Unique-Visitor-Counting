"""Entry point.
  python main.py             run on config.json source
  python main.py --report    print unique visitor count from DB
  python main.py --reset     wipe DB + logs, then run (clean demo run)
"""
import argparse
import glob
import os
import shutil
import time
import cv2

from src.config import load_config
from src.database import Database
from src.events_logger import setup_logger


def resolve_source(cfg, log):
    """Use configured source; if a local file is missing, fall back to first video in data/."""
    if cfg.is_stream or cfg.source.startswith(("rtsp://", "http")) or os.path.exists(cfg.source):
        return cfg.source
    vids = [f for ext in ("mp4", "avi", "mov", "mkv") for f in glob.glob(f"data/*.{ext}")]
    if not vids:
        raise SystemExit(f"No video found. Put the video in data/ (config source: {cfg.source})")
    log.warning("'%s' not found, using %s", cfg.source, vids[0])
    return vids[0]


def open_capture(cfg, log):
    src = resolve_source(cfg, log)
    while True:
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        if not cfg.is_stream:
            raise SystemExit(f"Cannot open source: {src}")
        log.warning("Stream unavailable, retrying in %ss", cfg.reconnect_delay_sec)
        time.sleep(cfg.reconnect_delay_sec)


def draw(frame, tracks):
    for t in tracks:
        x1, y1, x2, y2 = t.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {t.face_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.reset:
        shutil.rmtree(cfg.log_dir, ignore_errors=True)
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(cfg.db_path + suffix):
                os.remove(cfg.db_path + suffix)

    log = setup_logger(cfg.log_dir)
    db = Database(cfg.db_path)
    if args.report:
        print(f"Unique visitors: {db.unique_count()}  events: {db.event_counts()}")
        return

    from src.pipeline import FaceCounterPipeline
    pipe = FaceCounterPipeline(cfg, db)
    cap = open_capture(cfg, log)
    log.info("Processing | frame_skip=%d", cfg.frame_skip)
    idx, tracks = 0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if cfg.is_stream:
                    log.warning("Stream dropped; reconnecting")
                    cap.release(); cap = open_capture(cfg, log); continue
                break                                   # end of video file
            idx += 1
            if idx % (cfg.frame_skip + 1) == 0:         # detection cycle
                tracks = pipe.process(frame, idx)
            if cfg.display:
                draw(frame, tracks)
                cv2.putText(frame, f"Unique: {db.unique_count()}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Face Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        pipe.close()
        cap.release(); db.close(); cv2.destroyAllWindows()
    print("\nDone. Run: python main.py --report")


if __name__ == "__main__":
    main()
