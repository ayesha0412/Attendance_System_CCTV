"""Probe Hikvision RTSP channels and report actual stream quality.

Usage:
    python scripts/probe_stream_quality.py
"""

import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv


def build_url(username: str, password: str, ip: str, port: str, channel: str) -> str:
    return f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"


def probe_channel(url: str, seconds: float = 2.0) -> tuple:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False, 0, 0, 0.0, 0.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    frames = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if ret and frame is not None:
            frames += 1

    cap.release()
    eff_fps = frames / seconds if seconds > 0 else 0.0
    return True, w, h, meta_fps, eff_fps


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        print(f".env not found: {env_path}")
        return 1

    load_dotenv(env_path)

    username = os.getenv("CAMERA_USERNAME", "admin")
    password = os.getenv("CAMERA_PASSWORD", "")
    ip = os.getenv("CAMERA_IP", "")
    port = os.getenv("CAMERA_PORT", "554")

    if not ip:
        print("CAMERA_IP is empty in .env")
        return 1

    channels = ["101", "102", "201", "202"]
    print(f"Probing camera {ip}:{port} channels: {channels}")

    best = None
    for ch in channels:
        url = build_url(username, password, ip, port, ch)
        ok, w, h, meta_fps, eff_fps = probe_channel(url)
        if not ok:
            print(f"{ch}: open failed")
            continue

        pixels = w * h
        print(f"{ch}: {w}x{h} | meta_fps={meta_fps:.1f} | eff_fps={eff_fps:.1f}")
        if best is None or pixels > best[1]:
            best = (ch, pixels, w, h, meta_fps, eff_fps)

    if best is None:
        print("No channels opened successfully.")
        return 2

    print("\nRecommended channel for max quality:")
    print(f"{best[0]} ({best[2]}x{best[3]}, meta_fps={best[4]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
