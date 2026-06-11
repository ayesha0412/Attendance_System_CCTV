"""
Configuration Loader
====================
Merges config.yaml settings with .env secrets.
Single source of truth for all recognition settings.
"""

import os
import logging
import logging.handlers
import yaml
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # Recognition/
_PROJECT = os.path.dirname(_ROOT)       # Face_Detection_CCTV/


def load_config(config_path=None):
    """
    Load config.yaml and inject environment variables for secrets.

    Returns a plain dict — access with cfg["detection"]["score_min"] etc.
    """
    if config_path is None:
        config_path = os.path.join(_ROOT, "config.yaml")

    # Load .env from project root (camera creds, API keys)
    load_dotenv(os.path.join(_PROJECT, ".env"))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Camera credentials from .env ─────────────────────────────────
    cfg.setdefault("camera", {})
    cfg["camera"]["username"] = os.getenv("CAMERA_USERNAME", "super")
    cfg["camera"]["password"] = os.getenv("CAMERA_PASSWORD", "")
    cfg["camera"]["ip"]       = os.getenv("CAMERA_IP", "172.16.16.84")
    cfg["camera"]["port"]     = os.getenv("CAMERA_PORT", "554")
    cfg["camera"]["channel"]  = os.getenv("CAMERA_CHANNEL", "101")

    # ── API keys from .env ───────────────────────────────────────────
    cfg.setdefault("api", {})
    cfg["api"]["attendance_url"] = os.getenv("ATTENDANCE_API_URL", "")
    cfg["api"]["attendance_key"] = os.getenv("ATTENDANCE_API_KEY", "")
    cfg["api"]["read_key"]       = os.getenv("ATTENDANCE_READ_KEY", "")

    return cfg


def build_rtsp_url(cfg):
    """Construct RTSP URL from config camera section."""
    cam = cfg["camera"]
    u, p  = cam["username"], cam["password"]
    ip    = cam["ip"]
    port  = cam["port"]
    ch    = cam["channel"]
    return f"rtsp://{u}:{p}@{ip}:{port}/Streaming/Channels/{ch}"


def setup_logging(cfg):
    """Configure root logger — console + rotating file."""
    log_cfg = cfg.get("logging", {})
    level   = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    # Ensure log directory exists
    log_file = os.path.join(_ROOT, log_cfg.get("file", "logs/recognition.log"))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    # Rotating file handler
    file_h = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("max_bytes", 10_485_760),
        backupCount=log_cfg.get("backup_count", 5),
    )
    file_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_h)

    # Silence noisy third-party loggers
    logging.getLogger("insightface").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
