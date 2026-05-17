"""
Live Monitoring Dashboard — Flask + SocketIO.

Streams annotated video frames and face crops to the browser in real-time.
"""

import base64
import logging
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, render_template, Response, send_from_directory
from flask_socketio import SocketIO

logger = logging.getLogger("face_collector.dashboard")


class Dashboard:
    """
    Real-time dashboard server.

    Provides:
    - MJPEG live stream of annotated frames (faces boxed)
    - SocketIO events for new face captures (thumbnail + stats)
    - REST endpoint for face gallery
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000,
                 max_gallery_size: int = 50):
        self.host = host
        self.port = port
        self.max_gallery_size = max_gallery_size

        # Flask app setup
        template_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"
        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir),
        )
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")

        # Shared state
        self._annotated_frame = None
        self._frame_lock = threading.Lock()
        self._stats = {
            "faces_today": 0,
            "faces_total": 0,
            "fps": 0.0,
            "active_tracks": 0,
            "camera_connected": False,
            "skipped_blur": 0,
        }

        self._setup_routes()

    def _setup_routes(self):
        """Register Flask routes."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.route("/api/stats")
        def get_stats():
            return self._stats

    def _generate_frames(self):
        """MJPEG stream generator."""
        while True:
            with self._frame_lock:
                if self._annotated_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self._annotated_frame.copy()

            # Encode to JPEG
            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )
            time.sleep(0.033)  # ~30 fps max for dashboard

    def update_frame(self, annotated_frame):
        """Update the live stream frame (called from main loop)."""
        with self._frame_lock:
            self._annotated_frame = annotated_frame

    def notify_new_face(self, crop_image, track_id: int):
        """Emit SocketIO event when a new face is captured."""
        # Encode crop to base64 for browser display
        ret, buffer = cv2.imencode(".jpg", crop_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            b64 = base64.b64encode(buffer).decode("utf-8")
            self.socketio.emit("new_face", {
                "image": b64,
                "track_id": track_id,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    def update_stats(self, faces_today: int, faces_total: int, fps: float,
                     active_tracks: int, camera_connected: bool, skipped_blur: int = 0):
        """Update dashboard statistics."""
        self._stats = {
            "faces_today": faces_today,
            "faces_total": faces_total,
            "fps": round(fps, 1),
            "active_tracks": active_tracks,
            "camera_connected": camera_connected,
            "skipped_blur": skipped_blur,
        }
        self.socketio.emit("stats_update", self._stats)

    def start(self):
        """Start the dashboard server in a background thread."""
        def run():
            logger.info(f"Dashboard starting on http://{self.host}:{self.port}")
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info("Dashboard thread started")
        return thread


def create_dashboard(config: dict) -> Dashboard:
    """Factory function to create a Dashboard from config dict."""
    dash_cfg = config.get("dashboard", {})
    return Dashboard(
        host=dash_cfg.get("host", "0.0.0.0"),
        port=dash_cfg.get("port", 5000),
        max_gallery_size=dash_cfg.get("max_gallery_size", 50),
    )
