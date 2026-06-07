"""
Live Face Recognition Dashboard — Flask + MJPEG
=================================================
Loads the gallery built by train_embeddings.py, connects to the RTSP
camera, and classifies faces using cosine similarity against ArcFace
embeddings.  Temporal voting over a rolling window stabilises labels
so single blurry frames can't flip the displayed identity.

Usage:
    python live_recognition.py

Then open: http://localhost:5001
"""

import os
import sys
import cv2
import time
import pickle
import threading
import numpy as np
from collections import deque, Counter
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from flask import Flask, Response, render_template_string
from attendance_sender import mark_attendance

# =====================================================================
#  SETTINGS
# =====================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH          = os.path.join(_HERE, "face_model.pkl")
COSINE_THRESHOLD    = 0.42   # below this → Unknown  (tune 0.38–0.55)
DET_SCORE_MIN       = 0.75   # InsightFace detection confidence — drop below this
FACE_SIZE_MIN       = 80     # minimum face width AND height in pixels — raised to reject tiny noisy faces
FACE_SIZE_MAX       = 300    # reject faces larger than this — catches phones held close
DEBUG_SCORES        = True   # print cosine scores to terminal — set False once tuned
LIVENESS_CHECK      = True   # anti-spoof: reject screen/print texture
DET_SIZE            = (640, 640)
ENHANCE_LIVE_FRAME  = True   # CLAHE on live frame to normalise CCTV lighting
DASHBOARD_PORT      = 5001

# Temporal voting — how many frames to accumulate before committing an identity
VOTE_WINDOW         = 15     # rolling buffer length  (~1 sec at 15 fps)
VOTE_THRESHOLD      = 9      # need 9/15 (60 %) agreement to commit

# Sidebar log cooldown per person (seconds)
LOG_COOLDOWN_SECONDS = 15

# Centroid tracker — max pixel distance to match a face between frames
TRACKER_MAX_DIST    = 150    # pixels in original frame coords
TRACKER_MAX_AGE     = 20     # frames before a lost track is dropped

# =====================================================================
#  COLORS (BGR)
# =====================================================================
GREEN  = (0, 220, 0)
RED    = (0, 60, 220)
GRAY   = (140, 140, 140)
WHITE  = (255, 255, 255)

# =====================================================================
#  HTML TEMPLATE
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Face Recognition</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Inter',sans-serif; background:#0a0e17; color:#e2e8f0; min-height:100vh; }
        .app { max-width:1400px; margin:0 auto; padding:20px; }
        .header {
            display:flex; justify-content:space-between; align-items:center;
            padding:16px 24px;
            background:linear-gradient(135deg,#1a1f2e,#141824);
            border:1px solid #2a3040; border-radius:12px; margin-bottom:20px;
        }
        .header h1 {
            font-size:1.3rem; font-weight:700;
            background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        }
        .status { display:flex; align-items:center; gap:8px; font-size:0.85rem; color:#94a3b8; }
        .status-dot {
            width:10px; height:10px; border-radius:50%;
            background:#22c55e; box-shadow:0 0 8px #22c55e88;
            animation:pulse 2s infinite;
        }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }

        .main { display:grid; grid-template-columns:1fr 320px; gap:20px; }
        @media(max-width:900px){ .main{grid-template-columns:1fr} }

        .video-panel { background:#141824; border:1px solid #2a3040; border-radius:12px; overflow:hidden; }
        .panel-header {
            display:flex; justify-content:space-between; align-items:center;
            padding:12px 20px; border-bottom:1px solid #2a3040;
            font-weight:600; font-size:.95rem;
        }
        .badge { background:#22c55e22; color:#22c55e; padding:4px 12px; border-radius:20px; font-size:.8rem; font-weight:600; }
        .video-container { width:100%; background:#000; }
        .video-container img { width:100%; display:block; }

        .sidebar { display:flex; flex-direction:column; gap:16px; }
        .info-card { background:#141824; border:1px solid #2a3040; border-radius:12px; padding:20px; }
        .info-card h3 { font-size:.9rem; color:#94a3b8; margin-bottom:12px; text-transform:uppercase; letter-spacing:.05em; }
        .class-list { display:flex; flex-wrap:wrap; gap:6px; }
        .class-tag { background:#1e293b; color:#94a3b8; padding:5px 10px; border-radius:6px; font-size:.78rem; border:1px solid #334155; }

        .stats-bar { display:flex; gap:16px; margin-top:20px; flex-wrap:wrap; }
        .stat { flex:1; min-width:140px; background:#141824; border:1px solid #2a3040; border-radius:12px; padding:16px; text-align:center; }
        .stat-label { font-size:.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }
        .stat-value { font-size:1.5rem; font-weight:700; margin-top:4px; color:#e2e8f0; }
        .stat-value.green{color:#22c55e} .stat-value.blue{color:#60a5fa} .stat-value.yellow{color:#eab308}

        .log-container { max-height:400px; overflow-y:auto; }
        .log-container::-webkit-scrollbar{width:6px}
        .log-container::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
        .log-entry { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1e293b; font-size:.85rem; }
        .log-name { font-weight:600; color:#e2e8f0; }
        .log-conf { padding:2px 8px; border-radius:10px; font-size:.75rem; font-weight:600; }
        .log-conf.high{background:#22c55e22;color:#22c55e} .log-conf.low{background:#ef444422;color:#ef4444}
        .log-time { color:#64748b; font-size:.75rem; }
        .footer-note { text-align:center; color:#475569; font-size:.75rem; margin-top:16px; padding:8px; }
    </style>
</head>
<body>
<div class="app">
    <header class="header">
        <h1>Live Face Recognition</h1>
        <div class="status"><div class="status-dot"></div><span>LIVE — RTSP Stream</span></div>
    </header>

    <div class="main">
        <div class="video-panel">
            <div class="panel-header">
                <span>Annotated Feed</span>
                <span class="badge" id="fps-badge">-- FPS</span>
            </div>
            <div class="video-container">
                <img src="/video_feed" alt="Live Feed">
            </div>
        </div>

        <div class="sidebar">
            <div class="info-card">
                <h3>Registered Employees</h3>
                <div class="class-list">
                    {% for name in class_names %}
                    <span class="class-tag">{{ name }}</span>
                    {% endfor %}
                </div>
            </div>
            <div class="info-card">
                <h3>Employee Sightings</h3>
                <p style="font-size:.75rem;color:#475569;margin-bottom:8px;">
                    Logged once every {{ cooldown }}s per person.
                    Unknown faces counted in stats only.
                </p>
                <div id="log-container" class="log-container">
                    <div style="color:#64748b;font-size:.85rem;text-align:center;padding:20px;">
                        Waiting for recognised employees...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="stats-bar">
        <div class="stat"><div class="stat-label">Recognised</div><div class="stat-value green" id="s-rec">0</div></div>
        <div class="stat"><div class="stat-label">Unknown</div><div class="stat-value yellow" id="s-unk">0</div></div>
        <div class="stat"><div class="stat-label">Total Detections</div><div class="stat-value blue" id="s-tot">0</div></div>
        <div class="stat"><div class="stat-label">Uptime</div><div class="stat-value" id="s-up">00:00</div></div>
    </div>

    <div class="footer-note">
        Cosine threshold: {{ threshold }} &nbsp;|&nbsp;
        Vote window: {{ vote_window }} frames &nbsp;|&nbsp;
        Port: {{ port }}
    </div>
</div>

<script>
setInterval(() => {
    fetch('/api/stats').then(r => r.json()).then(data => {
        document.getElementById('s-rec').innerText = data.recognized;
        document.getElementById('s-unk').innerText = data.unknown;
        document.getElementById('s-tot').innerText = data.total;
        document.getElementById('fps-badge').innerText = data.fps.toFixed(1) + ' FPS';
        let m = Math.floor(data.uptime/60), s = Math.floor(data.uptime%60);
        document.getElementById('s-up').innerText =
            String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');

        const el = document.getElementById('log-container');
        if (data.recent_detections && data.recent_detections.length > 0) {
            el.innerHTML = data.recent_detections.map(d => {
                const cls = d.confidence >= 50 ? 'high' : 'low';
                const parts = d.time.split(' ');
                return `<div class="log-entry">
                    <span class="log-name">${d.name}</span>
                    <span class="log-conf ${cls}">${d.confidence}%</span>
                    <span class="log-time" title="${parts[0]||''}">${parts[1]||d.time}</span>
                </div>`;
            }).join('');
        }
    }).catch(()=>{});
}, 2000);
</script>
</body>
</html>
"""


# =====================================================================
#  CENTROID TRACKER
#  Associates detected face bboxes across frames by centroid distance.
#  Returns a stable track_id per face so temporal voting works correctly.
# =====================================================================
class CentroidTracker:
    def __init__(self, max_dist=TRACKER_MAX_DIST, max_age=TRACKER_MAX_AGE):
        self._tracks   = {}    # {tid: {"cx": float, "cy": float, "age": int}}
        self._next_id  = 0
        self.max_dist  = max_dist
        self.max_age   = max_age

    def update(self, bboxes):
        """
        bboxes: list of [x1, y1, x2, y2] in original frame coords.
        Returns: list of track_ids aligned with bboxes.
        """
        # Age all existing tracks
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        if not bboxes:
            return []

        centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes]
        assigned  = []
        used_tids = set()

        for cx, cy in centroids:
            best_tid, best_dist = None, self.max_dist

            for tid, t in self._tracks.items():
                if tid in used_tids:
                    continue
                dist = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is not None:
                self._tracks[best_tid].update({"cx": cx, "cy": cy, "age": 0})
                used_tids.add(best_tid)
            else:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = {"cx": cx, "cy": cy, "age": 0}

            assigned.append(best_tid)

        return assigned


# =====================================================================
#  THREADED CAMERA
# =====================================================================
class ThreadedCamera:
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock    = threading.Lock()
        self.frame   = None
        self.running = False

    def start(self):
        if not self.cap.isOpened():
            return False
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        return True

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.frame = frame if ret else self.frame

    def read(self):
        with self.lock:
            return (self.frame is not None), (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.running = False
        self.cap.release()


# =====================================================================
#  GLOBALS
# =====================================================================
annotated_frame = None
frame_lock      = threading.Lock()
stats = {
    "recognized": 0, "unknown": 0, "total": 0,
    "fps": 0.0, "uptime": 0.0, "recent_detections": [],
}
stats_lock       = threading.Lock()
MAX_LOG_ENTRIES  = 50
_log_cooldown: dict = {}    # {name: last_log_timestamp}
_model_info      = (None, [])  # (gallery_dict, class_names)


# =====================================================================
#  CLAHE PRE-PROCESSING — normalise CCTV lighting before embedding
# =====================================================================
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_frame(frame):
    """Apply CLAHE to the L channel to even out IR / uneven CCTV lighting."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# =====================================================================
#  LIVENESS / ANTI-SPOOF
# =====================================================================
def check_liveness(face_bgr):
    """Return True if the face appears real, False if likely a screen or print."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.log1p(np.abs(np.fft.fftshift(f)))
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    high_freq = magnitude.copy()
    high_freq[ch - 10:ch + 10, cw - 10:cw + 10] = 0
    freq_ratio = np.sum(high_freq) / (np.sum(magnitude) + 1e-8)
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_fake = freq_ratio > 0.85 and sat_std < 25 and lap_var < 100
    if is_fake and DEBUG_SCORES:
        print(f"[SPOOF] freq={freq_ratio:.2f} sat_std={sat_std:.1f} lap={lap_var:.1f} -> REJECTED")
    return not is_fake


# =====================================================================
#  COSINE CLASSIFY
# =====================================================================
def cosine_classify(embedding, gallery):
    """
    L2-normalise the query embedding, dot-product against each gallery
    entry (which are pre-normalised), return (name, score).
    Score range: -1.0 to 1.0  — same person typically 0.45–0.80 on CCTV.
    """
    emb = embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    scores = {name: float(np.dot(emb, ref)) for name, ref in gallery.items()}
    best_name  = max(scores, key=scores.get)
    best_score = scores[best_name]

    if DEBUG_SCORES:
        score_str = "  ".join(f"{n}: {s:.2f}" for n, s in sorted(scores.items()))
        result    = best_name if best_score >= COSINE_THRESHOLD else "Unknown"
        print(f"[SCORE] {score_str}  →  {result}")

    if best_score < COSINE_THRESHOLD:
        best_name = "Unknown"

    return best_name, best_score


# =====================================================================
#  DRAWING
# =====================================================================
def draw_box(frame, bbox, label, score, state):
    """
    state: 'known' | 'unknown' | 'identifying'
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = GREEN if state == "known" else (RED if state == "unknown" else GRAY)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cl = max(12, int((x2 - x1) * 0.15))
    t  = 3
    for (ax, ay), (bx, by) in [
        ((x1,y1),(x1+cl,y1)), ((x1,y1),(x1,y1+cl)),
        ((x2,y1),(x2-cl,y1)), ((x2,y1),(x2,y1+cl)),
        ((x1,y2),(x1+cl,y2)), ((x1,y2),(x1,y2-cl)),
        ((x2,y2),(x2-cl,y2)), ((x2,y2),(x2,y2-cl)),
    ]:
        cv2.line(frame, (ax, ay), (bx, by), color, t)

    if state == "identifying":
        text = "Identifying..."
    else:
        text = f"{label}  {score:.0%}"

    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _  = cv2.getTextSize(text, font, fs, ft)
    ly1 = max(0, y1 - th - 12)
    ly2 = ly1 + th + 12
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 10, ly2), color, -1)
    cv2.putText(frame, text, (x1 + 5, ly2 - 4), font, fs, WHITE, ft, cv2.LINE_AA)


# =====================================================================
#  HELPERS
# =====================================================================
def load_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Model not found: {path}")
        print("        Run: python train_embeddings.py")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "gallery" not in data:
        print("[ERROR] face_model.pkl is the old SVM format.")
        print("        Re-run: python train_embeddings.py")
        sys.exit(1)
    print(f"[OK] Gallery loaded — {len(data['gallery'])} people: {data['class_names']}")
    return data["gallery"], data["class_names"]


def build_rtsp_url():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE, "..", ".env"))
    u = os.getenv("CAMERA_USERNAME", "super")
    p = os.getenv("CAMERA_PASSWORD", "pakistan123")
    i = os.getenv("CAMERA_IP",       "172.16.16.84")
    o = os.getenv("CAMERA_PORT",     "554")
    c = os.getenv("CAMERA_CHANNEL",  "101")
    print(f"[OK] RTSP: rtsp://{u}:****@{i}:{o}/Streaming/Channels/{c}")
    return f"rtsp://{u}:{p}@{i}:{o}/Streaming/Channels/{c}"


def init_face_analyzer():
    print("[..] Loading InsightFace (buffalo_l)...")
    fa = FaceAnalysis(name="buffalo_l",
                      providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=DET_SIZE)
    print("[OK] InsightFace ready!\n")
    return fa


# =====================================================================
#  INFERENCE THREAD
# =====================================================================
def inference_loop(camera, face_app, gallery):
    global annotated_frame, stats

    tracker          = CentroidTracker()
    vote_buffers     = {}    # {track_id: deque(maxlen=VOTE_WINDOW)}
    committed_names  = {}    # {track_id: last committed name}  for log dedup

    start_time  = time.time()
    frame_times = []

    while camera.running:
        t0 = time.time()
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # --- Enhance frame to normalise CCTV lighting ---
        if ENHANCE_LIVE_FRAME:
            frame = enhance_frame(frame)

        # --- Run detection at full CCTV resolution ---
        faces = face_app.get(frame)

        # --- Filter detections ---
        # det_score < DET_SCORE_MIN  → background false positive (texture, poster, reflection)
        # face too small             → too far away or a spurious blob, not worth recognising
        faces = [
            f for f in faces
            if f.det_score >= DET_SCORE_MIN
            and FACE_SIZE_MIN <= (f.bbox[2] - f.bbox[0]) <= FACE_SIZE_MAX
            and FACE_SIZE_MIN <= (f.bbox[3] - f.bbox[1]) <= FACE_SIZE_MAX
        ]

        # --- Bboxes already in original resolution ---
        bboxes = [face.bbox for face in faces]

        # --- Assign track IDs via centroid matching ---
        track_ids = tracker.update(bboxes)

        new_log_entries = []
        recognized = 0
        unknown    = 0
        now        = time.time()

        for face, bbox, tid in zip(faces, bboxes, track_ids):

            # --- Liveness check on face crop ---
            if LIVENESS_CHECK:
                x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and not check_liveness(crop):
                    draw_box(frame, bbox, "SPOOF", 0.0, "unknown")
                    continue

            # --- Cosine classify this single frame ---
            raw_name, raw_score = cosine_classify(face.embedding, gallery)

            # --- Push raw result into this track's vote buffer ---
            if tid not in vote_buffers:
                vote_buffers[tid] = deque(maxlen=VOTE_WINDOW)
            vote_buffers[tid].append((raw_name, raw_score))

            buf = vote_buffers[tid]

            # --- Decide what to display ---
            if len(buf) < 5:
                # Not enough frames yet — show "Identifying..."
                draw_box(frame, bbox, "", 0.0, "identifying")
                continue

            # Majority vote across the buffer
            voted_name, vote_count = Counter(n for n, _ in buf).most_common(1)[0]
            # Median of top scores for the winning name — ignores bad frames
            winner_scores = sorted([s for n, s in buf if n == voted_name], reverse=True)
            voted_score = float(np.median(winner_scores[:max(3, len(winner_scores)//2)]))

            if vote_count < VOTE_THRESHOLD:
                # No clear winner yet — show leading candidate as uncertain
                draw_box(frame, bbox, voted_name, voted_score, "identifying")
                continue

            # --- Committed identity ---
            is_known = (voted_name != "Unknown")
            state    = "known" if is_known else "unknown"
            draw_box(frame, bbox, voted_name, voted_score, state)

            if is_known:
                recognized += 1
            else:
                unknown += 1

            # --- Sidebar log: only when identity commits or changes,
            #     subject to per-person cooldown ---
            prev = committed_names.get(tid)
            if voted_name != prev:
                committed_names[tid] = voted_name

            if is_known:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                mark_attendance(voted_name, ts)
                last_logged = _log_cooldown.get(voted_name, 0)
                if now - last_logged >= LOG_COOLDOWN_SECONDS:
                    _log_cooldown[voted_name] = now
                    new_log_entries.append({
                        "name":       voted_name,
                        "confidence": int(voted_score * 100),
                        "time":       ts,
                    })

        # --- FPS ---
        frame_times.append(time.time() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

        with frame_lock:
            annotated_frame = frame

        with stats_lock:
            stats["recognized"] += recognized
            stats["unknown"]    += unknown
            stats["total"]      += recognized + unknown
            stats["fps"]         = round(fps, 1)
            stats["uptime"]      = time.time() - start_time
            if new_log_entries:
                stats["recent_detections"] = (
                    new_log_entries + stats["recent_detections"]
                )[:MAX_LOG_ENTRIES]


# =====================================================================
#  FLASK APP
# =====================================================================
app = Flask(__name__)


def generate_mjpeg():
    while True:
        with frame_lock:
            if annotated_frame is None:
                time.sleep(0.05)
                continue
            frame = annotated_frame.copy()

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    gallery, class_names = _model_info
    return render_template_string(
        HTML_TEMPLATE,
        class_names=class_names,
        threshold=COSINE_THRESHOLD,
        cooldown=LOG_COOLDOWN_SECONDS,
        vote_window=VOTE_WINDOW,
        port=DASHBOARD_PORT,
    )


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def get_stats():
    with stats_lock:
        return dict(stats)


# =====================================================================
#  MAIN
# =====================================================================
def main():
    global _model_info

    print("=" * 55)
    print("  LIVE FACE RECOGNITION — Flask Dashboard")
    print("=" * 55)
    print()

    gallery, class_names = load_model(MODEL_PATH)
    _model_info          = (gallery, class_names)

    face_app = init_face_analyzer()
    rtsp_url = build_rtsp_url()

    print("[..] Connecting to camera...")
    camera = ThreadedCamera(rtsp_url)
    if not camera.start():
        print("[ERROR] Cannot connect to camera!")
        sys.exit(1)
    print("[OK] Camera connected!\n")

    time.sleep(1)

    threading.Thread(
        target=inference_loop,
        args=(camera, face_app, gallery),
        daemon=True,
    ).start()
    print("[OK] Inference thread started\n")
    print(f"[OK] Dashboard → http://localhost:{DASHBOARD_PORT}\n")

    try:
        app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        print("\n[OK] Stopped.")


if __name__ == "__main__":
    main()
