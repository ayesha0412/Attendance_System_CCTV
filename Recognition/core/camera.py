"""
Threaded RTSP Camera Reader
============================
Reads frames in a background thread so the inference loop never blocks on I/O.
Provides thread-safe frame access via read().
"""

import cv2
import logging
import threading

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Continuously grabs frames from an RTSP (or any OpenCV) source in a
    daemon thread.  The main loop calls ``read()`` to get the latest frame."""

    def __init__(self, url, buffer_size=1):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.lock    = threading.Lock()
        self.frame   = None
        self.running = False

    def start(self):
        """Open the stream and spawn the reader thread.  Returns True on success."""
        if not self.cap.isOpened():
            logger.error("Cannot open camera stream")
            return False
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        logger.info("Camera reader thread started")
        return True

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.frame = frame if ret else self.frame

    def read(self):
        """Return (success, frame).  Frame is a copy — safe to mutate."""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()
        logger.info("Camera stopped")
