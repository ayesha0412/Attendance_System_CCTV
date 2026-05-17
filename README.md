# Attendance_System_CCTV

Interactive pipeline for CCTV face data collection, augmentation, and live attendance recognition.

## Jump To

- [Project Overview](#project-overview)
- [Pipeline Stages](#pipeline-stages)
- [Quick Start](#quick-start)
- [Run Commands](#run-commands)
- [Directory Guide](#directory-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Project Overview

This repository is organized as a staged workflow:

1. Stage 1 collects faces from CCTV/RTSP streams and stores raw crops by date and camera.
2. Stage 2 augments approved faces for stronger training data.
3. Stage 3 trains and runs live recognition for attendance.

## Pipeline Stages

| Stage | Goal | Main Folder | Launcher |
|---|---|---|---|
| 1 | Collect raw face crops from CCTV | `Data_Collection/` | `stage1_collect.bat` |
| 2 | Data augmentation for training robustness | `Data_Augmentation/` | `stage2_augment.bat` |
| 3 | Train and run recognition | `Recognition/` | `stage3_train.bat`, `stage3_live.bat` |
| 3 (AdaFace) | AdaFace training + live recognition | `Recognition/` | `stage3_adaface_setup.bat`, `stage3_adaface_train.bat`, `stage3_adaface_live.bat` |

<details>
<summary>What makes this pipeline effective?</summary>

- RTSP + tracker-based capture to reduce duplicate saves
- Quality filtering to skip blurred face crops
- Date/camera folder strategy for easy traceability
- Separate data collection and recognition stages for maintainability

</details>

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ayesha0412/Attendance_System_CCTV.git
cd Attendance_System_CCTV
```

### 2. Stage Environments

Each stage has its own dependency file:

- `Data_Collection/requirements.txt`
- `Recognition/requirements.txt`

Install per stage in your preferred virtual environment/conda environment.

### 3. Configure Collection

Update camera settings in `Data_Collection/config.yaml` and set credentials in local environment variables or `.env` (do not commit secrets).

### 4. Run by Stage

Use the `.bat` launchers from the project root.

## Run Commands

From project root:

```bat
stage1_collect.bat
stage2_augment.bat
stage3_train.bat
stage3_live.bat
```

AdaFace flow:

```bat
stage3_adaface_setup.bat
stage3_adaface_train.bat
stage3_adaface_live.bat
```

## Directory Guide

```text
Face_Detection_CCTV/
|- Data_Collection/
|  |- src/
|  |- scripts/
|  |- data/raw_faces/YYYY-MM-DD/camera_XX/
|- Data_Augmentation/
|  |- augment_faces.py
|  |- Data_Gathering/
|  |- Augmented_Data/
|- Recognition/
|  |- train_embeddings.py
|  |- live_recognition.py
|  |- train_adaface.py
|  |- live_adaface.py
|- models/
|- logs/
|- stage1_collect.bat
|- stage2_augment.bat
|- stage3_train.bat
|- stage3_live.bat
|- stage3_adaface_setup.bat
|- stage3_adaface_train.bat
|- stage3_adaface_live.bat
```

## Configuration

Main runtime config: `Data_Collection/config.yaml`

Common parameters to tune:

- Camera: source URL, reconnect behavior, FPS cap
- Detection: confidence threshold, minimum face size
- Tracking: capture interval and association behavior
- Storage: save path, image quality, blur threshold

## Troubleshooting

1. If stream fails, verify RTSP URL and camera credentials.
2. If detection is slow, reduce input size/FPS and check GPU runtime.
3. If push fails due to large files, keep model binaries out of git or use Git LFS.
4. If no faces are saved, lower detection threshold and verify face size constraints.

## Roadmap

- Multi-camera orchestration dashboard
- Automated hard-negative filtering
- Improved attendance analytics and alerts
- Optional model version tracking

## License

Add your preferred license (MIT/Apache-2.0) in a `LICENSE` file.
