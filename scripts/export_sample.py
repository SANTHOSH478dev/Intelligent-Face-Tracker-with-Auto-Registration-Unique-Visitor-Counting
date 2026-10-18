"""Collect submission evidence into sample_output/ (events.log, DB dumps, sample images)."""
import csv, glob, os, shutil, sqlite3
from src.config import load_config

cfg = load_config()
out = "sample_output"; os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(cfg.log_dir, "events.log"), out)
conn = sqlite3.connect(cfg.db_path)
for table in ("faces", "events"):
    cur = conn.execute(f"SELECT * FROM {table}" if table == "events"
                       else "SELECT face_id, first_seen, image_path FROM faces")
    with open(os.path.join(out, f"{table}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow([c[0] for c in cur.description]); w.writerows(cur)
with open(os.path.join(out, "summary.txt"), "w") as f:
    f.write(f"Unique visitors: {conn.execute('SELECT COUNT(*) FROM faces').fetchone()[0]}\n")
    f.write(str(dict(conn.execute('SELECT event_type, COUNT(*) FROM events GROUP BY 1').fetchall())))
for kind in ("entries", "exits", "faces"):
    imgs = sorted(glob.glob(os.path.join(cfg.log_dir, kind, "*", "*.jpg")))[:6]
    os.makedirs(os.path.join(out, "images", kind), exist_ok=True)
    for i in imgs:
        shutil.copy(i, os.path.join(out, "images", kind))
print("Sample output saved in sample_output/")
