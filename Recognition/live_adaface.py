"""
Live Face Recognition — AdaFace
=================================
Uses InsightFace SCRFD for detection + landmark extraction,
aligns faces to 112x112, then classifies with AdaFace IR-50
embeddings via cosine similarity gallery.

Usage:
    python live_adaface.py

Dashboard: http://localhost:5002  (different port from ArcFace at 5001)
"""

import os
import sys
import cv2
import time
import pickle
import threading
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque, Counter
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from flask import Flask, Response, render_template_string

# =====================================================================
#  SETTINGS
# =====================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR          = os.path.join(_HERE, "adaface_assets")
MODEL_DIR           = os.path.join(ASSETS_DIR, "cvlface_model")
MODEL_PATH          = os.path.join(_HERE, "face_model_adaface.pkl")
COSINE_THRESHOLD    = 0.42
DET_SCORE_MIN       = 0.75
FACE_SIZE_MIN       = 60
DET_SIZE            = (320, 320)
PROCESS_SCALE       = 0.5
DASHBOARD_PORT      = 5002       # separate port so both ArcFace + AdaFace can run together
VOTE_WINDOW         = 15
VOTE_THRESHOLD      = 9
LOG_COOLDOWN_SECONDS = 15
TRACKER_MAX_DIST    = 150
TRACKER_MAX_AGE     = 20
DEBUG_SCORES        = True       # set False once happy with results

# =====================================================================
#  COLORS (BGR)
# =====================================================================
GREEN = (0, 220, 0)
RED   = (0, 60, 220)
GRAY  = (140, 140, 140)
WHITE = (255, 255, 255)

# =====================================================================
#  HTML (same dashboard, just shows AdaFace in footer)
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdaFace Recognition</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#0a0e17;color:#e2e8f0;min-height:100vh}
.app{max-width:1400px;margin:0 auto;padding:20px}
.header{display:flex;justify-content:space-between;align-items:center;padding:16px 24px;
  background:linear-gradient(135deg,#1a1f2e,#141824);border:1px solid #2a3040;border-radius:12px;margin-bottom:20px}
.header h1{font-size:1.3rem;font-weight:700;background:linear-gradient(135deg,#f59e0b,#ef4444);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.status{display:flex;align-items:center;gap:8px;font-size:.85rem;color:#94a3b8}
.status-dot{width:10px;height:10px;border-radius:50%;background:#f59e0b;box-shadow:0 0 8px #f59e0b88;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.main{display:grid;grid-template-columns:1fr 320px;gap:20px}
@media(max-width:900px){.main{grid-template-columns:1fr}}
.video-panel{background:#141824;border:1px solid #2a3040;border-radius:12px;overflow:hidden}
.panel-header{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;
  border-bottom:1px solid #2a3040;font-weight:600;font-size:.95rem}
.badge{background:#f59e0b22;color:#f59e0b;padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:600}
.video-container{width:100%;background:#000}
.video-container img{width:100%;display:block}
.sidebar{display:flex;flex-direction:column;gap:16px}
.info-card{background:#141824;border:1px solid #2a3040;border-radius:12px;padding:20px}
.info-card h3{font-size:.9rem;color:#94a3b8;margin-bottom:12px;text-transform:uppercase;letter-spacing:.05em}
.class-list{display:flex;flex-wrap:wrap;gap:6px}
.class-tag{background:#1e293b;color:#94a3b8;padding:5px 10px;border-radius:6px;font-size:.78rem;border:1px solid #334155}
.stats-bar{display:flex;gap:16px;margin-top:20px;flex-wrap:wrap}
.stat{flex:1;min-width:140px;background:#141824;border:1px solid #2a3040;border-radius:12px;padding:16px;text-align:center}
.stat-label{font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.stat-value{font-size:1.5rem;font-weight:700;margin-top:4px;color:#e2e8f0}
.stat-value.green{color:#22c55e}.stat-value.blue{color:#60a5fa}.stat-value.yellow{color:#eab308}
.log-container{max-height:400px;overflow-y:auto}
.log-container::-webkit-scrollbar{width:6px}
.log-container::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
.log-entry{display:flex;justify-content:space-between;align-items:center;padding:8px 0;
  border-bottom:1px solid #1e293b;font-size:.85rem}
.log-name{font-weight:600;color:#e2e8f0}
.log-conf{padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600}
.log-conf.high{background:#22c55e22;color:#22c55e}.log-conf.low{background:#ef444422;color:#ef4444}
.log-time{color:#64748b;font-size:.75rem}
.footer-note{text-align:center;color:#475569;font-size:.75rem;margin-top:16px;padding:8px}
</style></head><body>
<div class="app">
  <header class="header">
    <h1>AdaFace Recognition</h1>
    <div class="status"><div class="status-dot"></div><span>LIVE — AdaFace IR-50</span></div>
  </header>
  <div class="main">
    <div class="video-panel">
      <div class="panel-header">
        <span>Annotated Feed</span>
        <span class="badge" id="fps-badge">-- FPS</span>
      </div>
      <div class="video-container"><img src="/video_feed" alt="Live Feed"></div>
    </div>
    <div class="sidebar">
      <div class="info-card">
        <h3>Registered Employees</h3>
        <div class="class-list">
          {% for name in class_names %}<span class="class-tag">{{ name }}</span>{% endfor %}
        </div>
      </div>
      <div class="info-card">
        <h3>Employee Sightings</h3>
        <p style="font-size:.75rem;color:#475569;margin-bottom:8px">
          Logged once every {{ cooldown }}s per person.</p>
        <div id="log-container" class="log-container">
          <div style="color:#64748b;font-size:.85rem;text-align:center;padding:20px">
            Waiting for recognised employees...</div>
        </div>
      </div>
    </div>
  </div>
  <div class="stats-bar">
    <div class="stat"><div class="stat-label">Recognised</div><div class="stat-value green" id="s-rec">0</div></div>
    <div class="stat"><div class="stat-label">Unknown</div><div class="stat-value yellow" id="s-unk">0</div></div>
    <div class="stat"><div class="stat-label">Total</div><div class="stat-value blue" id="s-tot">0</div></div>
    <div class="stat"><div class="stat-label">Uptime</div><div class="stat-value" id="s-up">00:00</div></div>
  </div>
  <div class="footer-note">
    Model: AdaFace IR-50 MS1MV3 &nbsp;|&nbsp;
    Threshold: {{ threshold }} &nbsp;|&nbsp;
    Vote: {{ vote_window }} frames &nbsp;|&nbsp;
    Port: {{ port }}
  </div>
</div>
<script>
setInterval(()=>{
  fetch('/api/stats').then(r=>r.json()).then(d=>{
    document.getElementById('s-rec').innerText=d.recognized;
    document.getElementById('s-unk').innerText=d.unknown;
    document.getElementById('s-tot').innerText=d.total;
    document.getElementById('fps-badge').innerText=d.fps.toFixed(1)+' FPS';
    let m=Math.floor(d.uptime/60),s=Math.floor(d.uptime%60);
    document.getElementById('s-up').innerText=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
    const el=document.getElementById('log-container');
    if(d.recent_detections&&d.recent_detections.length>0){
      el.innerHTML=d.recent_detections.map(x=>{
        const cls=x.confidence>=50?'high':'low';
        const parts=x.time.split(' ');
        return`<div class="log-entry">
          <span class="log-name">${x.name}</span>
          <span class="log-conf ${cls}">${x.confidence}%</span>
          <span class="log-time" title="${parts[0]||''}">${parts[1]||x.time}</span></div>`;
      }).join('');
    }
  }).catch(()=>{});
},2000);
</script></body></html>
"""


# =====================================================================
#  FACE ALIGNMENT  (112x112 standard for ArcFace / AdaFace)
# =====================================================================
_LANDMARK_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_112(img, kps):
    M, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32).reshape(-1, 1, 2),
        _LANDMARK_DST.reshape(-1, 1, 2),
    )
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


# =====================================================================
#  ADAFACE EMBEDDING
# =====================================================================
_adaface_model  = None
_adaface_device = "cpu"


def load_adaface_model():
    global _adaface_model, _adaface_device

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"[ERROR] Model not found at {MODEL_DIR}")
        print("        Run: python setup_adaface.py")
        sys.exit(1)

    _prev_cwd = os.getcwd()
    try:
        os.chdir(MODEL_DIR)
        if MODEL_DIR not in sys.path:
            sys.path.insert(0, MODEL_DIR)
        from models.iresnet import load_model as _load_iresnet
        from omegaconf import OmegaConf
        import yaml
        conf = OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml")))
        model = _load_iresnet(conf)
        model.load_state_dict_from_path("pretrained_model/model.pt")
    finally:
        os.chdir(_prev_cwd)

    model.eval()

    # RTX 5080 (Blackwell SM 12.0) needs PyTorch nightly for CUDA kernel support.
    # Run a tiny forward pass on CUDA to confirm compatibility; fall back to CPU otherwise.
    _adaface_device = "cpu"
    if torch.cuda.is_available():
        try:
            model.cuda()
            with torch.no_grad():
                model(torch.zeros(1, 3, 112, 112).cuda())
            _adaface_device = "cuda"
        except Exception:
            model.cpu()
            _adaface_device = "cpu"
            print("[WARN] CUDA not supported for this GPU by current PyTorch — AdaFace on CPU.")

    _adaface_model = model
    print(f"[OK] AdaFace IR-50 on {_adaface_device.upper()}")


def get_embedding(aligned_bgr):
    img    = aligned_bgr[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img    = (img / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(_adaface_device)
    with torch.no_grad():
        output = _adaface_model(tensor)   # IResNetModel.forward(x) — plain tensor
    if isinstance(output, (list, tuple)):
        emb = output[0]
    else:
        emb = output
    if emb.dim() > 2:
        emb = emb[:, 0]
    emb = F.normalize(emb, dim=1)
    return emb.squeeze().cpu().numpy()


# =====================================================================
#  COSINE CLASSIFY
# =====================================================================
def cosine_classify(embedding, gallery):
    emb    = embedding.astype(np.float32)
    emb    = emb / (np.linalg.norm(emb) + 1e-8)
    scores = {n: float(np.dot(emb, r)) for n, r in gallery.items()}

    best_name  = max(scores, key=scores.get)
    best_score = scores[best_name]

    if DEBUG_SCORES:
        s = "  ".join(f"{n}: {v:.2f}" for n, v in sorted(scores.items()))
        print(f"[ADA] {s}  →  {best_name if best_score >= COSINE_THRESHOLD else 'Unknown'}")

    if best_score < COSINE_THRESHOLD:
        best_name = "Unknown"
    return best_name, best_score


# =====================================================================
#  CENTROID TRACKER  (same as live_recognition.py)
# =====================================================================
class CentroidTracker:
    def __init__(self):
        self._tracks  = {}
        self._next_id = 0

    def update(self, bboxes):
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > TRACKER_MAX_AGE:
                del self._tracks[tid]

        if not bboxes:
            return []

        centroids = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes]
        assigned, used = [], set()

        for cx, cy in centroids:
            best_tid, best_dist = None, TRACKER_MAX_DIST
            for tid, t in self._tracks.items():
                if tid in used:
                    continue
                d = ((cx-t["cx"])**2 + (cy-t["cy"])**2)**0.5
                if d < best_dist:
                    best_dist, best_tid = d, tid

            if best_tid is not None:
                self._tracks[best_tid].update({"cx": cx, "cy": cy, "age": 0})
                used.add(best_tid)
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
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
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
stats = {"recognized":0,"unknown":0,"total":0,"fps":0.0,"uptime":0.0,"recent_detections":[]}
stats_lock      = threading.Lock()
MAX_LOG_ENTRIES = 50
_log_cooldown: dict = {}
_model_info     = (None, [])


# =====================================================================
#  DRAWING
# =====================================================================
def draw_box(frame, bbox, label, score, state):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    color = GREEN if state=="known" else (RED if state=="unknown" else GRAY)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cl = max(12,int((x2-x1)*0.15)); t=3
    for (ax,ay),(bx,by) in [
        ((x1,y1),(x1+cl,y1)),((x1,y1),(x1,y1+cl)),
        ((x2,y1),(x2-cl,y1)),((x2,y1),(x2,y1+cl)),
        ((x1,y2),(x1+cl,y2)),((x1,y2),(x1,y2-cl)),
        ((x2,y2),(x2-cl,y2)),((x2,y2),(x2,y2-cl)),
    ]: cv2.line(frame,(ax,ay),(bx,by),color,t)
    text = "Identifying..." if state=="identifying" else f"{label}  {score:.0%}"
    font,fs,ft = cv2.FONT_HERSHEY_SIMPLEX,0.6,2
    (tw,th),_ = cv2.getTextSize(text,font,fs,ft)
    ly1=max(0,y1-th-12); ly2=ly1+th+12
    cv2.rectangle(frame,(x1,ly1),(x1+tw+10,ly2),color,-1)
    cv2.putText(frame,text,(x1+5,ly2-4),font,fs,WHITE,ft,cv2.LINE_AA)


# =====================================================================
#  INFERENCE THREAD
# =====================================================================
def inference_loop(camera, detector, gallery):
    global annotated_frame, stats

    tracker         = CentroidTracker()
    vote_buffers    = {}
    committed_names = {}
    start_time      = time.time()
    frame_times     = []

    while camera.running:
        t0 = time.time()
        ret, frame = camera.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        small     = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                               interpolation=cv2.INTER_LINEAR)
        faces     = detector.get(small)
        scale_inv = 1.0 / PROCESS_SCALE

        # Filter by detection score and face size
        faces = [
            f for f in faces
            if f.det_score >= DET_SCORE_MIN
            and (f.bbox[2]-f.bbox[0]) * scale_inv >= FACE_SIZE_MIN
            and (f.bbox[3]-f.bbox[1]) * scale_inv >= FACE_SIZE_MIN
            and f.kps is not None
        ]

        bboxes    = [f.bbox * scale_inv for f in faces]
        track_ids = tracker.update(bboxes)

        new_log_entries = []
        recognized = 0
        unknown    = 0
        now        = time.time()

        for face, bbox, tid in zip(faces, bboxes, track_ids):
            # Scale landmarks back to original resolution and align
            kps     = face.kps * scale_inv
            aligned = align_112(frame, kps)
            if aligned is None:
                continue

            emb              = get_embedding(aligned)
            raw_name, raw_score = cosine_classify(emb, gallery)

            if tid not in vote_buffers:
                vote_buffers[tid] = deque(maxlen=VOTE_WINDOW)
            vote_buffers[tid].append((raw_name, raw_score))
            buf = vote_buffers[tid]

            if len(buf) < 5:
                draw_box(frame, bbox, "", 0.0, "identifying")
                continue

            voted_name, vote_count = Counter(n for n,_ in buf).most_common(1)[0]
            voted_score = float(np.mean([s for n,s in buf if n==voted_name]))

            if vote_count < VOTE_THRESHOLD:
                draw_box(frame, bbox, voted_name, voted_score, "identifying")
                continue

            is_known = voted_name != "Unknown"
            draw_box(frame, bbox, voted_name, voted_score,
                     "known" if is_known else "unknown")

            if is_known:
                recognized += 1
                last_logged = _log_cooldown.get(voted_name, 0)
                if now - last_logged >= LOG_COOLDOWN_SECONDS:
                    _log_cooldown[voted_name] = now
                    new_log_entries.append({
                        "name":       voted_name,
                        "confidence": int(voted_score * 100),
                        "time":       time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
            else:
                unknown += 1

        frame_times.append(time.time()-t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times)/len(frame_times)) if frame_times else 0

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
#  FLASK
# =====================================================================
app = Flask(__name__)


def generate_mjpeg():
    while True:
        with frame_lock:
            if annotated_frame is None:
                time.sleep(0.05); continue
            frame = annotated_frame.copy()
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret: continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(0.033)


@app.route("/")
def index():
    _, class_names = _model_info
    return render_template_string(HTML_TEMPLATE,
        class_names=class_names,
        threshold=COSINE_THRESHOLD,
        cooldown=LOG_COOLDOWN_SECONDS,
        vote_window=VOTE_WINDOW,
        port=DASHBOARD_PORT)

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stats")
def get_stats():
    with stats_lock:
        return dict(stats)


# =====================================================================
#  MAIN
# =====================================================================
def build_rtsp_url():
    load_dotenv(os.path.join(_HERE, "..", ".env"))
    u = os.getenv("CAMERA_USERNAME","super")
    p = os.getenv("CAMERA_PASSWORD","pakistan123")
    i = os.getenv("CAMERA_IP",      "172.16.16.84")
    o = os.getenv("CAMERA_PORT",    "554")
    c = os.getenv("CAMERA_CHANNEL", "101")
    print(f"[OK] RTSP: rtsp://{u}:****@{i}:{o}/Streaming/Channels/{c}")
    return f"rtsp://{u}:{p}@{i}:{o}/Streaming/Channels/{c}"


def main():
    global _model_info

    print("="*55)
    print("  LIVE RECOGNITION — AdaFace IR-50")
    print("="*55); print()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run: python train_adaface.py")
        sys.exit(1)

    with open(MODEL_PATH,"rb") as f:
        data = pickle.load(f)
    gallery     = data["gallery"]
    class_names = data["class_names"]
    _model_info = (gallery, class_names)
    print(f"[OK] Gallery loaded — {len(gallery)} people: {class_names}\n")

    load_adaface_model()

    print("\n[..] Loading InsightFace SCRFD (detection only)...")
    detector = FaceAnalysis(name="buffalo_l",
                            providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=DET_SIZE)
    print("[OK] Detector ready.\n")

    rtsp_url = build_rtsp_url()
    print("[..] Connecting to camera...")
    camera = ThreadedCamera(rtsp_url)
    if not camera.start():
        print("[ERROR] Cannot connect to camera!"); sys.exit(1)
    print("[OK] Camera connected.\n")

    time.sleep(1)

    threading.Thread(target=inference_loop,
                     args=(camera, detector, gallery), daemon=True).start()
    print("[OK] Inference started.\n")
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
