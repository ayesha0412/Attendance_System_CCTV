"""
Face Data Collector — Pipeline Orchestrator.

Connects all modules: camera → detector → tracker → saver (→ dashboard).
This is the core processing loop that ties everything together.
"""

import time
import logging

import cv2
import numpy as np

from .camera import RTSPStream
from .detector import FaceDetector
from .tracker import FaceTracker
from .storage import ImageSaver

logger = logging.getLogger("face_collector")


class FaceCollector:
    """
    Main pipeline orchestrator.

    Grabs frames from camera, detects faces, tracks them,
    saves new/cooldown-expired faces, and feeds the dashboard.
    """

    def __init__(self, config: dict):
        self.config = config
        self._running = False
        self._dashboard = None
        self.camera_id = config.get("camera", {}).get("camera_id", "camera_01")

        # Initialize modules from config
        cam_cfg = config["camera"]
        det_cfg = config["detection"]
        trk_cfg = config["tracker"]
        sto_cfg = config["storage"]

        self.camera = RTSPStream(
            rtsp_url=cam_cfg["rtsp_url"],
            fps_cap=cam_cfg.get("fps_cap", 15),
            reconnect_delay=cam_cfg.get("reconnect_delay", 2),
            max_reconnect_delay=cam_cfg.get("max_reconnect_delay", 30),
            transport=cam_cfg.get("transport", "tcp"),
        )
        self.process_fps = max(1, int(cam_cfg.get("fps_cap", 10)))
        self.process_interval = 1.0 / self.process_fps
        self._last_process_ts = 0.0

        self.detector = FaceDetector(
            model_path=det_cfg.get("model_path", None),
            model_name=det_cfg.get("model_name", "buffalo_l"),
            confidence=det_cfg.get("confidence", 0.5),
            min_face_size=det_cfg.get("min_face_size", 80),
            gpu_id=det_cfg.get("gpu_id", 0),
            input_size=tuple(det_cfg.get("input_size", [640, 640])),
            max_frame_side=det_cfg.get("max_frame_side", 960),
        )

        self.tracker = FaceTracker(
            iou_threshold=trk_cfg.get("iou_threshold", 0.4),
            cooldown_seconds=trk_cfg.get("capture_interval_seconds", 2),
            max_faces=trk_cfg.get("max_faces", 20),
            stale_timeout=trk_cfg.get("stale_timeout", 30),
            backend=trk_cfg.get("backend", "bytetrack"),
        )

        self.saver = ImageSaver(
            output_dir=sto_cfg.get("output_dir", "data/raw_faces"),
            image_size=tuple(sto_cfg.get("image_size", [224, 224])),
            quality=sto_cfg.get("quality", 95),
            blur_threshold=sto_cfg.get("blur_threshold", 50.0),
            save_all_captures=sto_cfg.get("save_all_captures", False),
            padding=sto_cfg.get("padding", 0.3),
            max_queue_size=sto_cfg.get("max_queue_size", 0),
        )

        # FPS tracking
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = time.time()

    def set_dashboard(self, dashboard):
        """Attach a dashboard instance for live monitoring."""
        self._dashboard = dashboard

    def initialize(self):
        """Initialize all modules (call before run)."""
        logger.info("=" * 50)
        logger.info("Initializing Face Data Collector")
        logger.info("=" * 50)

        # Load face detection model (downloads on first run)
        self.detector.initialize()
        logger.info("Face detector ready")

        # Start background threads
        self.saver.start()
        logger.info("Image saver ready")

        self.camera.start()
        logger.info("Camera stream starting...")

        # Wait for first frame
        logger.info("Waiting for camera connection...")
        for _ in range(100):  # Wait up to 10 seconds
            if self.camera.is_connected:
                break
            time.sleep(0.1)

        if self.camera.is_connected:
            logger.info("Camera connected!")
        else:
            logger.warning("Camera not connected yet — will keep trying in background")

        logger.info("=" * 50)
        logger.info("Collector initialized. Starting processing...")
        logger.info("=" * 50)

    def run(self):
        """
        Main processing loop.
        Press Ctrl+C to stop.
        """
        self._running = True

        while self._running:
            # Grab latest frame
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - self._last_process_ts < self.process_interval:
                # Decode thread already keeps freshest frame; this keeps compute bounded.
                time.sleep(0.001)
                continue
            self._last_process_ts = now

            # Detect faces
            faces = self.detector.detect(frame)

            # Log frame resolution once
            if self._frame_count == 0:
                fh, fw = frame.shape[:2]
                logger.info(f"Processing frame resolution: {fw}x{fh}")

            # Track faces and get saveable ones
            faces_to_save = self.tracker.update(faces)

            # Save new/cooldown-expired faces
            for track_id, face in faces_to_save:
                queued = self.saver.save(frame, face.bbox, track_id, self.camera_id,
                                         landmarks=face.landmarks)
                if not queued:
                    continue

                # Notify dashboard of new face
                if self._dashboard is not None:
                    crop = self._crop_face(frame, face.bbox)
                    if crop is not None:
                        self._dashboard.notify_new_face(crop, track_id)

            # Draw detections on frame for dashboard
            annotated = self.detector.draw_detections(frame, faces)

            # Send annotated frame to dashboard
            if self._dashboard is not None:
                self._dashboard.update_frame(annotated)

            # Update FPS counter
            self._frame_count += 1
            elapsed = time.time() - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = time.time()

                # Update dashboard stats
                if self._dashboard is not None:
                    self._dashboard.update_stats(
                        faces_today=self.saver.get_today_count(),
                        faces_total=self.saver.saved_count,
                        fps=self._fps,
                        active_tracks=self.tracker.active_tracks,
                        camera_connected=self.camera.is_connected,
                        skipped_blur=self.saver.skipped_blur_count,
                    )

        logger.info("Processing loop ended")

    def stop(self):
        """Gracefully stop all modules."""
        logger.info("Shutting down collector...")
        self._running = False
        self.camera.stop()
        self.saver.stop()
        logger.info("Collector stopped")

    @staticmethod
    def _crop_face(frame: np.ndarray, bbox: np.ndarray,
                   padding: float = 0.2) -> np.ndarray:
        """Quick crop for dashboard thumbnail."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (160, 160))

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.stop()