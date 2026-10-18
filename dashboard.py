"""Live dashboard (bonus frontend):  python dashboard.py  ->  http://localhost:5000"""
import os
import sqlite3
from flask import Flask, jsonify, send_file, abort, Response
from src.config import load_config

cfg = load_config()
app = Flask(__name__)
LOG_ROOT = os.path.abspath(cfg.log_dir)


def q(sql):
    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def img_url(p):
    return "/img/" + p.replace("\\", "/") if p else None


@app.route("/api/data")
def data():
    try:
        unique = q("SELECT COUNT(*) c FROM faces")[0]["c"]
        counts = {r["event_type"]: r["c"] for r in
                  q("SELECT event_type, COUNT(*) c FROM events GROUP BY event_type")}
        events = q("SELECT event_id, face_id, event_type, timestamp, image_path "
                   "FROM events ORDER BY event_id DESC LIMIT 30")
        faces = q("SELECT face_id, first_seen, image_path FROM faces ORDER BY face_id DESC LIMIT 24")
    except sqlite3.OperationalError:
        unique, counts, events, faces = 0, {}, [], []
    for r in events + faces:
        r["image"] = img_url(r.pop("image_path"))
    entries, exits = counts.get("entry", 0), counts.get("exit", 0)
    return jsonify(unique=unique, entries=entries, exits=exits,
                   inside=max(entries - exits, 0), events=events, faces=faces)


@app.route("/img/<path:p>")
def img(p):
    full = os.path.abspath(p)
    if not full.startswith(LOG_ROOT) or not os.path.isfile(full):
        abort(404)
    return send_file(full)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Face Tracker Dashboard</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
h1{margin:0 0 4px}.sub{color:#94a3b8;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.card{background:#1e293b;border-radius:12px;padding:18px}
.n{font-size:40px;font-weight:700}.l{color:#94a3b8}
.green{color:#4ade80}.red{color:#f87171}.blue{color:#60a5fa}.yellow{color:#facc15}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}
.panel{background:#1e293b;border-radius:12px;padding:16px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:8px;border-bottom:1px solid #334155;font-size:14px}
td img{height:44px;border-radius:6px}
.badge{padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600}
.entry{background:#14532d;color:#4ade80}.exit{background:#7f1d1d;color:#f87171}
.faces{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.faces div{text-align:center;font-size:12px;color:#94a3b8}.faces img{width:100%;border-radius:8px}
</style></head><body>
<h1>Intelligent Face Tracker</h1>
<div class="sub">Live unique visitor counter &middot; refreshes every 3 seconds</div>
<div class="cards">
 <div class="card"><div class="n blue" id="unique">0</div><div class="l">Unique Visitors</div></div>
 <div class="card"><div class="n green" id="entries">0</div><div class="l">Total Entries</div></div>
 <div class="card"><div class="n red" id="exits">0</div><div class="l">Total Exits</div></div>
 <div class="card"><div class="n yellow" id="inside">0</div><div class="l">Currently In Frame</div></div>
</div>
<div class="grid">
 <div class="panel"><h3>Recent Events</h3><table><thead><tr><th>Image</th><th>Face ID</th><th>Event</th><th>Time</th></tr></thead><tbody id="events"></tbody></table></div>
 <div class="panel"><h3>Registered Faces</h3><div class="faces" id="faces"></div></div>
</div>
<script>
async function load(){
  const d = await (await fetch('/api/data')).json();
  for (const k of ['unique','entries','exits','inside']) document.getElementById(k).textContent = d[k];
  document.getElementById('events').innerHTML = d.events.map(e =>
    '<tr><td>' + (e.image ? '<img src="' + e.image + '">' : '') + '</td><td>ID ' + e.face_id +
    '</td><td><span class="badge ' + e.event_type + '">' + e.event_type.toUpperCase() +
    '</span></td><td>' + e.timestamp.replace('T',' ').slice(0,19) + '</td></tr>').join('');
  document.getElementById('faces').innerHTML = d.faces.map(f =>
    '<div>' + (f.image ? '<img src="' + f.image + '">' : '') + 'ID ' + f.face_id + '</div>').join('');
}
load(); setInterval(load, 3000);
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
