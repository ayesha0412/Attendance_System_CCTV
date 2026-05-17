"""
Image Saver — crops, aligns, enhances, and saves detected faces.

Runs a background thread with a save queue so the
main detection loop is never blocked by disk I/O.

Quality pipeline:
  1. Landmark-based alignment (eyes horizontal, face centered)
  2. CLAHE contrast enhancement (even out CCTV lighting)
  3. Blur rejection
  4. High-quality PNG output
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Full
from typing import Optional

import cv2
import numpy as np

from ..utils.helpers import compute_blur, align_face, enhance_face

logger = logging.getLogger("face_collector.storage")


class ImageSaver:
    """
    Threaded face image saver with alignment, enhancement, and blur detection.

    Usage:
        saver = ImageSaver(output_dir="data/raw_faces", image_size=(320, 320))
        saver.start()
        saver.save(frame, bbox, track_id, landmarks=landmarks)
        ...
        saver.stop()
    """

    def __init__(self, output_dir: str = "data/raw_faces",
                 image_size: tuple = (320, 320),
                 quality: int = 98,
                 blur_threshold: float = 50.0,
                 save_all_captures: bool = False,
                 padding: float = 0.4,
                 max_queue_size: int = 0):
        self.output_dir = Path(output_dir)
        self.image_size = tuple(image_size)
        self.quality = quality
        self.blur_threshold = blur_threshold
        self.save_all_captures = save_all_captures
        self.padding = padding

        # max_queue_size=0 means unbounded queue in Python's Queue.
        self._queue = Queue(maxsize=max(0, int(max_queue_size)))
        self._running = False
        self._thread = None
        self._saved_count = 0
        self._skipped_blur = 0
        self._latest_crop = None
        self._lock = threading.Lock()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Include existing files so dashboard totals are not reset to zero each run.
        self._saved_count = self.get_total_count()

    @property
    def saved_count(self) -> int:
        return self._saved_count

    @property
    def skipped_blur_count(self) -> int:
        return self._skipped_blur

    @property
    def latest_crop(self) -> Optional[np.ndarray]:
        """Get the most recently saved face crop (for dashboard)."""
        with self._lock:
            return self._latest_crop.copy() if self._latest_crop is not None else None

    def start(self):
        """Start the background save thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._thread.start()
        logger.info(f"Image saver started — output: {self.output_dir}")

    def stop(self):
        """Stop the save thread (finishes current queue)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
        logger.info(f"Image saver stopped. Saved: {self._saved_count}, Skipped (blur): {self._skipped_blur}")

    def save(self, frame: np.ndarray, bbox: np.ndarray, track_id: int,
             camera_id: str = "camera_01", landmarks: np.ndarray = None):
        """
        Queue a face crop for saving.
        Non-blocking — drops the save if queue is full.
        """
        try:
            lm = landmarks.copy() if landmarks is not None else None
            self._queue.put_nowait((frame.copy(), bbox.copy(), track_id, camera_id, lm))
            return True
        except Exception as exc:
            logger.warning("Failed to queue save request: %s", exc)
            return False

    def _save_loop(self):
        """Background loop that processes the save queue."""
        while self._running or not self._queue.empty():
            try:
                frame, bbox, track_id, camera_id, landmarks = self._queue.get(timeout=1.0)
            except Exception:
                continue

            self._process_save(frame, bbox, track_id, camera_id, landmarks)

    def _process_save(self, frame: np.ndarray, bbox: np.ndarray, track_id: int,
                      camera_id: str, landmarks: np.ndarray = None):
        """Align, enhance, validate, and save a single face."""
        h, w = frame.shape[:2]

        # ---- Step 1: Aligned crop using landmarks ----
        aligned = None
        if landmarks is not None:
            aligned = align_face(frame, bbox, landmarks,
                                 output_size=self.image_size,
                                 padding=self.padding)

        if aligned is not None:
            crop = aligned
        else:
            # Fallback: simple bbox crop with padding
            x1, y1, x2, y2 = bbox
            face_w = x2 - x1
            face_h = y2 - y1
            pad_x = int(face_w * self.padding)
            pad_y = int(face_h * self.padding)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop.size == 0:
                return
            crop = cv2.resize(raw_crop, self.image_size, interpolation=cv2.INTER_LANCZOS4)

        # ---- Step 2: Blur check ----
        blur_score = compute_blur(crop)
        if (not self.save_all_captures) and blur_score < self.blur_threshold:
            self._skipped_blur += 1
            logger.debug(f"Skipped blurry face (score: {blur_score:.1f} < {self.blur_threshold})")
            return

        # ---- Step 3: CLAHE contrast enhancement ----
        enhanced = enhance_face(crop)

        # ---- Step 4: Save to disk ----
        now = datetime.now()
        date_dir = self.output_dir / now.strftime("%Y-%m-%d") / camera_id
        date_dir.mkdir(parents=True, exist_ok=True)

        base_name = (
            f"{now.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            f"_{camera_id}_t{track_id:04d}_{self._saved_count:06d}"
        )

        # Save as PNG (lossless — best for training)
        filepath = date_dir / f"{base_name}.png"
        cv2.imwrite(str(filepath), enhanced)

        self._saved_count += 1

        # Store latest crop for dashboard display
        with self._lock:
            self._latest_crop = enhanced

        logger.debug(f"Saved face: {filepath} (blur: {blur_score:.1f}, "
                     f"aligned={'yes' if aligned is not None else 'no'})")

    def get_today_count(self) -> int:
        """Count face images saved today."""
        today_dir = self.output_dir / datetime.now().strftime("%Y-%m-%d")
        if not today_dir.exists():
            return 0
        return len(list(today_dir.glob("**/*.png")) + list(today_dir.glob("**/*.jpg")))

    def get_total_count(self) -> int:
        """Count all face images ever saved."""
        return len(list(self.output_dir.glob("**/*.png")) + list(self.output_dir.glob("**/*.jpg")))

    def get_recent_crops(self, n: int = 10) -> list:
        """Get paths to the N most recently saved face images."""
        imgs = sorted(
            list(self.output_dir.glob("**/*.png")) + list(self.output_dir.glob("**/*.jpg")),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        return imgs[:n]
