"""
Utility helpers — config loading, env loading, logging setup.
"""

import os
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_env(env_path: str = None) -> dict:
    """
    Load camera credentials from .env file.
    Looks in the project root (two levels up from Data_Collection/src/).
    Returns a dict with RTSP URL and individual variables.
    """
    if env_path is None:
        # Navigate from Data_Collection/ → project root
        project_root = Path(__file__).resolve().parents[3]
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        raise FileNotFoundError(
            f".env file not found at {env_path}\n"
            f"Create it from the template with your Hikvision credentials."
        )

    load_dotenv(env_path)

    username = os.getenv("CAMERA_USERNAME", "admin")
    password = os.getenv("CAMERA_PASSWORD", "")
    ip = os.getenv("CAMERA_IP", "192.168.1.64")
    port = os.getenv("CAMERA_PORT", "554")
    # Hikvision main stream is typically channel 101 (higher resolution)
    channel = os.getenv("CAMERA_CHANNEL", "101")

    # Construct Hikvision RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"

    return {
        "rtsp_url": rtsp_url,
        "username": username,
        "ip": ip,
        "port": port,
        "channel": channel,
    }


def load_config(config_path: str = None) -> dict:
    """
    Load YAML config and merge with .env credentials.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load env and inject RTSP URL if not overridden in config
    env = load_env()
    if config.get("camera", {}).get("rtsp_url") is None:
        config["camera"]["rtsp_url"] = env["rtsp_url"]

    return config


def setup_logging(level: str = "INFO", log_dir: str = None) -> logging.Logger:
    """
    Configure project-wide logging.
    Logs to both console and file (if log_dir provided).
    """
    logger = logging.getLogger("face_collector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "collector.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def compute_blur(image) -> float:
    """
    Compute image blurriness using Laplacian variance.
    Higher value = sharper image.
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def align_face(frame: 'np.ndarray', bbox: 'np.ndarray', landmarks: 'np.ndarray',
               output_size: tuple = (320, 320), padding: float = 0.4) -> 'np.ndarray':
    """
    Crop and align a face using 5-point landmarks (2 eyes, nose, 2 mouth corners).

    Performs similarity transform to:
      1. Make the eye line horizontal
      2. Center the face in the output image
      3. Scale to fill the output size with padding

    This is the standard alignment used by ArcFace/CosFace recognition models.

    Args:
        frame: Full BGR frame
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks: 5×2 array of facial landmarks
        output_size: (width, height) of output crop
        padding: Extra padding ratio around the face

    Returns:
        Aligned face crop, or None if alignment fails
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    # Validate landmarks — need at least the two eye points
    if landmarks is None or landmarks.shape[0] < 2:
        return None
    if np.allclose(landmarks, 0):
        return None

    left_eye = landmarks[0]   # Left eye center
    right_eye = landmarks[1]  # Right eye center

    # Compute rotation angle from eye line
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Eye center as rotation pivot
    eye_center = ((left_eye[0] + right_eye[0]) / 2.0,
                  (left_eye[1] + right_eye[1]) / 2.0)

    # Face center (use bbox center for more stability)
    face_cx = (bbox[0] + bbox[2]) / 2.0
    face_cy = (bbox[1] + bbox[3]) / 2.0

    # Face size with padding
    face_w = (bbox[2] - bbox[0]) * (1 + padding)
    face_h = (bbox[3] - bbox[1]) * (1 + padding)
    face_size = max(face_w, face_h)  # Make it square

    # Scale factor to fit into output
    scale = output_size[0] / face_size if face_size > 0 else 1.0

    # Build rotation matrix around face center
    M = cv2.getRotationMatrix2D((float(face_cx), float(face_cy)), float(angle), float(scale))

    # Adjust translation so face center maps to output center
    M[0, 2] += output_size[0] / 2.0 - face_cx
    M[1, 2] += output_size[1] / 2.0 - face_cy

    # Apply affine warp
    aligned = cv2.warpAffine(
        frame, M, output_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


def enhance_face(crop: 'np.ndarray', clip_limit: float = 2.0,
                 tile_size: int = 8) -> 'np.ndarray':
    """
    Enhance face crop quality using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Improves visibility in uneven CCTV lighting without over-amplifying noise.

    Args:
        crop: BGR face crop
        clip_limit: CLAHE contrast limit (higher = more enhancement)
        tile_size: Grid size for local histogram equalization

    Returns:
        Enhanced BGR face crop
    """
    import cv2

    # Convert to LAB color space — only enhance L (lightness) channel
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced
