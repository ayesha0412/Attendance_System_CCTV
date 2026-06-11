# Face Recognition Attendance System (CCTV)

End-to-end pipeline for **CCTV-based employee attendance** using face recognition.
Captures faces from an RTSP stream, recognises them against a gallery of registered
employees, stores attendance in SQLite, and pushes events to an external AMS
(Attendance Management System) via REST.

> Two recognition backends are supported: **ArcFace** (InsightFace built-in, fast)
> and **AdaFace** (designed for low-quality CCTV, more accurate). They run side-by-side
> on different ports.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Repository Layout](#repository-layout)
4. [Database Schema](#database-schema)
5. [REST API](#rest-api)
6. [Quick Start](#quick-start)
7. [Adding a New Employee](#adding-a-new-employee)
8. [Configuration](#configuration)
9. [Tech Stack](#tech-stack)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
   Hikvision CCTV (RTSP) ─► SCRFD detect ─► AdaFace/ArcFace embed
                                                │
                                                ▼
                                       Cosine match vs gallery
                                                │
                                  ┌─────────────┴─────────────┐
                                  ▼                           ▼
                          SQLite (local)              External AMS
                          • attendance               (POST employee_id
                          • sightings                 + timestamp)
                                  │
                                  ▼
                            FastAPI / dashboard
                            (MJPEG stream + REST)
```

A single frame flows: **RTSP → CLAHE → SCRFD → liveness → embedding →
cosine vs gallery → vote buffer (15 frames) → log emp_no → DB + AMS push**.

---

## Pipeline Stages

| Stage | Goal | Folder | Launcher |
|---|---|---|---|
| 1 | Collect face crops from CCTV | `Data_Collection/` | `stage1_collect.bat` |
| 2 | Augment training data | `Data_Augmentation/` | `stage2_augment.bat` |
| 3a | Train **ArcFace** gallery | `Recognition/` | `stage3_train.bat` |
| 3b | Live recognition (ArcFace, port 5001) | `Recognition/` | `stage3_live.bat` |
| 3c | Train **AdaFace** gallery | `Recognition/` | `stage3_adaface_train.bat` |
| 3d | Live recognition (AdaFace, port 5002) | `Recognition/` | `stage3_adaface_live.bat` |
| — | Initial DB setup | `Recognition/` | `python migrate_db.py` |
| — | Add/update employees later | `Recognition/` | `python sync_employees.py` |

---

## Repository Layout

```
Face_Detection_CCTV/
├── .env                          # Camera + API secrets (gitignored)
├── .env.example                  # Template — copy to .env
├── README.md
│
├── stage1_collect.bat            # Capture from RTSP
├── stage2_augment.bat            # Augment training data
├── stage3_train.bat              # Build ArcFace gallery
├── stage3_live.bat               # Live recognition (ArcFace, :5001)
├── stage3_adaface_setup.bat      # Download AdaFace weights
├── stage3_adaface_train.bat      # Build AdaFace gallery
├── stage3_adaface_live.bat       # Live recognition (AdaFace, :5002)
│
├── Data_Collection/              # STAGE 1
│   ├── main.py
│   ├── config.yaml
│   ├── requirements.txt
│   └── src/
│       ├── camera/rtsp_stream.py        # Threaded RTSP reader
│       ├── detector/face_detector.py    # InsightFace SCRFD wrapper
│       ├── tracker/face_tracker.py      # ByteTrack-style tracker
│       ├── storage/image_saver.py       # Align + CLAHE + save
│       └── dashboard/app.py             # Capture preview UI
│
├── Data_Augmentation/            # STAGE 2
│   ├── augment_faces.py                 # Albumentations pipeline
│   ├── extract_faces_from_groups.py
│   ├── leplace_detection.py
│   ├── Dataset_Clicked/                 # Clean phone photos (gitignored)
│   └── Augmented_Data/                  # Augmented output (gitignored)
│
└── Recognition/                  # STAGE 3
    │
    ├── train_embeddings.py              # Build ArcFace gallery (.pkl)
    ├── train_adaface.py                 # Build AdaFace gallery (.pkl)
    ├── live_recognition.py              # Entry point — ArcFace (:5001)
    ├── live_adaface.py                  # Entry point — AdaFace (:5002)
    │
    ├── migrate_db.py                    # First-time DB setup from CSV
    ├── sync_employees.py                # Incremental employee sync
    │
    ├── config.yaml                      # Recognition tunables
    ├── requirements.txt
    ├── attendance.db                    # SQLite (gitignored)
    ├── employees.csv                    # Employee master list (gitignored)
    ├── face_model.pkl                   # ArcFace gallery (gitignored)
    ├── face_model_adaface.pkl           # AdaFace gallery (gitignored)
    │
    ├── adaface_assets/                  # AdaFace IR-50 model files
    │   ├── net.py
    │   └── cvlface_model/
    │
    ├── core/                            # Stateless building blocks
    │   ├── config.py                    # YAML + .env loader
    │   ├── camera.py                    # ThreadedCamera
    │   ├── alignment.py                 # 112×112 face alignment
    │   ├── classify.py                  # Cosine similarity matching
    │   ├── tracker.py                   # Centroid tracker
    │   ├── liveness.py                  # Anti-spoof (screen/print)
    │   ├── enhance.py                   # CLAHE lighting fix
    │   └── drawing.py                   # Bounding box rendering
    │
    ├── services/                        # Application services
    │   ├── recognition.py               # Main inference loop
    │   ├── attendance.py                # Mark attendance + push to AMS
    │   ├── database.py                  # SQLite layer
    │   └── state.py                     # Shared in-process state
    │
    ├── api/                             # FastAPI HTTP layer
    │   ├── __init__.py                  # App factory
    │   ├── middleware/auth.py           # Bearer token middleware
    │   └── blueprints/
    │       ├── dashboard.py             # / + /video_feed + /api/stats
    │       ├── attendance.py            # /api/v1/attendance/*
    │       └── health.py                # /api/v1/health
    │
    └── templates/
        └── dashboard.html               # Live MJPEG + sidebar UI
```

---

## Database Schema

Three tables in `Recognition/attendance.db`:

### `employees` — Master list (loaded from `employees.csv`)
| Column | Type | Description |
|---|---|---|
| emp_no | TEXT PRIMARY KEY | "061", "169" — leading zeros preserved |
| emp_name | TEXT | "Adnan Idrees Akhtar" |
| folder_name | TEXT UNIQUE | Must match `Dataset_Clicked/<folder>` |

### `attendance` — One row per employee per day
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| emp_no | TEXT | FK → employees |
| date | TEXT | "2026-06-11" |
| first_seen | TEXT | "09:15:32" — effective check-in |
| last_seen | TEXT | "17:42:11" — effective check-out (keeps updating) |
| confidence | INTEGER | Max confidence that day (0-100) |
| total_sightings | INTEGER | Total recognitions today |

**UNIQUE(emp_no, date)** — exactly one record per person per day.

### `sightings` — Every recognition event + AMS push queue
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| emp_no | TEXT | FK → employees |
| timestamp | TEXT | ISO 8601: "2026-06-11T09:15:32" |
| confidence | INTEGER | 0-100 |
| ams_sent | INTEGER | 0=pending, 1=POST attempted |
| ams_processed | INTEGER | 0=pending, 1=AMS returned HTTP 2xx |
| ams_attempts | INTEGER | Retry counter |
| ams_last_error | TEXT | Last failure message (truncated) |

A background worker retries `ams_sent=0` rows every 30 seconds.

---

## REST API

Auto-documented at `http://localhost:<port>/docs` (Swagger UI).

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | — | HTML dashboard with live MJPEG feed |
| GET | `/video_feed` | — | MJPEG annotated video stream |
| GET | `/api/stats` | — | Live counters (FPS, recognised, etc.) |
| GET | `/api/v1/health` | — | System status + uptime |
| GET | `/api/v1/attendance/today` | Bearer | Employees seen today |
| GET | `/api/v1/attendance/date/{date}` | Bearer | Specific date (YYYY-MM-DD) |
| GET | `/api/v1/attendance/employee/{emp_no}` | Bearer | One person's history (?days=N) |
| GET | `/api/v1/attendance/sightings` | Bearer | Recent events + AMS push status |

### AMS Push (outgoing)

When `ATTENDANCE_API_URL` is set in `.env`, every recognition event POSTs:

```json
POST <ATTENDANCE_API_URL>
Authorization: Bearer <ATTENDANCE_API_KEY>
Content-Type: application/json

{
  "employee_id": "061",
  "timestamp":   "2026-06-11T09:15:32"
}
```

The AMS system handles the semantics: first push of the day = check-in, all subsequent = check-out.

---

## Quick Start

### 1. Clone & install
```bash
git clone <repo-url>
cd Face_Detection_CCTV
```

Each stage has its own requirements:
```bash
pip install -r Data_Collection/requirements.txt
pip install -r Recognition/requirements.txt
```

### 2. Configure secrets
```bash
copy .env.example .env
# Edit .env with your camera IP/credentials and AMS API URL
```

### 3. Set up employees
1. Add rows to `Recognition/employees.csv` (`Employee ID, Employee Name, FOLDER_NAME`)
2. Drop 5-15 clean phone photos per employee in `Data_Augmentation/Dataset_Clicked/<FOLDER_NAME>/`
3. Initialise the database:
   ```bash
   cd Recognition
   python migrate_db.py
   ```

### 4. Download AdaFace model (one-time)
```bash
stage3_adaface_setup.bat
```

### 5. Train the gallery
```bash
stage3_adaface_train.bat       # AdaFace (recommended for CCTV)
# or
stage3_train.bat               # ArcFace
```

### 6. Run live
```bash
stage3_adaface_live.bat        # → http://localhost:5002
# or
stage3_live.bat                # → http://localhost:5001
```

---

## Adding a New Employee

```bash
# 1. Edit Excel → save as CSV (overwrites Recognition/employees.csv)
# 2. Create photo folder + drop in clean photos
mkdir Data_Augmentation\Dataset_Clicked\New Employee Name

# 3. Sync DB (preserves attendance history)
cd Recognition
python sync_employees.py

# 4. Re-train gallery
cd ..
stage3_adaface_train.bat
```

The new employee will be recognised the next time the live system starts.

---

## Configuration

### `.env` — Secrets (NOT committed)
Camera credentials, AMS API URL/key, and the read-key for protecting your REST API.
See `.env.example` for the full list.

### `Recognition/config.yaml` — Tunables
```yaml
detection:
  score_min: 0.65          # SCRFD confidence cutoff
  face_size_min: 60        # Min face size in pixels
  face_size_max: 300       # Reject phones held close to camera

recognition:
  cosine_threshold: 0.42   # Below this → "Unknown"
  vote_window: 15          # Rolling buffer (~1s at 15 fps)
  vote_threshold: 9        # Need 9/15 agreement to commit
  log_cooldown_seconds: 15 # Min seconds between logs per person
  liveness_check: true     # Anti-spoof (screens / printed photos)

preprocessing:
  enhance_clahe: false     # CLAHE lighting normalisation (slow, helps ArcFace)
```

### `Data_Collection/config.yaml`
Camera FPS cap, capture interval per face, blur threshold, output paths.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Camera | Hikvision CCTV via RTSP/H.264 |
| Detection | InsightFace SCRFD (`buffalo_l`) |
| Embedding | ArcFace R50 + AdaFace IR-50 (MS1MV3) |
| Liveness | Custom texture-based anti-spoof |
| Tracking | Centroid tracker |
| Backend | Python 3, FastAPI, Uvicorn |
| Database | SQLite (WAL mode) |
| Frontend | Jinja2 HTML + MJPEG streaming |
| Compute | NVIDIA CUDA with CPU fallback |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `attendance.db not found` | Run `python migrate_db.py` first |
| `No employees in DB` | Make sure `employees.csv` exists and has rows with `FOLDER_NAME` filled |
| RTSP stream won't connect | Check `.env` credentials, ping camera IP, try sub-stream (`CAMERA_CHANNEL=102`) |
| Recognition returns `Unknown` | Retrain gallery — gallery folder names must match `employees.csv → FOLDER_NAME` exactly |
| AMS push fails | Check `ATTENDANCE_API_URL`, watch `sightings.ams_last_error` in DB |
| CUDA error on RTX 5080 (Blackwell) | Falls back to CPU automatically; install PyTorch nightly for full GPU support |
| Low confidence scores (50-60%) | Add more varied photos per person, run `train_adaface.py`, ensure `face_size_min ≥ 80` |

---

## License

Add your preferred license (MIT / Apache 2.0) in a `LICENSE` file.
