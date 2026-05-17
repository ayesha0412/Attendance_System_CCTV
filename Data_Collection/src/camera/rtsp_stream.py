"""
Threaded RTSP stream grabber for Hikvision IP cameras.

Runs a background thread that continuously grabs frames from the RTSP
stream, ensuring the main processing loop always gets the latest frame
without blocking on network I/O.
"""

import time
import logging
import threading

import cv2

logger = logging.getLogger("face_collector.camera")


class RTSPStream:
    """
    Threaded RTSP video stream reader with auto-reconnect.

    Usage:
        stream = RTSPStream(rtsp_url, fps_cap=15)
        stream.start()
        while True:
            frame = stream.read()
            if frame is not None:
                # process frame
            ...
        stream.stop()
    """

    def __init__(self, rtsp_url: str, fps_cap: int = 15,
                 reconnect_delay: float = 2.0, max_reconnect_delay: float = 30.0,
                 transport: str = "tcp"):
        self.rtsp_url = rtsp_url
        self.fps_cap = fps_cap
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.transport = transport

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
        self._cap = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self):
        """Start the background frame grabber thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        logger.info("RTSP stream thread started")

    def stop(self):
        """Stop the background thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._cap is not None:
            self._cap.release()
        self._connected = False
        logger.info("RTSP stream stopped")

    def read(self):
        """
        Get the latest frame (thread-safe).
        Returns None if no frame is available.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _connect(self) -> bool:
        """Attempt to connect to the RTSP stream."""
        if self._cap is not None:
            self._cap.release()

        import os
        
        # Check if rtsp_url is an integer (webcam) or string (RTSP URL)
        if isinstance(self.rtsp_url, int):
            # Webcam input - use default backend
            self._cap = cv2.VideoCapture(self.rtsp_url)
        else:
            # Hikvision cameras work best with TCP transport
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}"
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Keep the decoder buffer as small as possible to reduce latency.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self._cap.isOpened():
            self._connected = True
            # Log the stream properties
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Connected to RTSP stream: {w}x{h} @ {fps:.1f} fps")
            return True
        else:
            self._connected = False
            logger.warning("Failed to connect to RTSP stream")
            return False

    def _grab_loop(self):
        """
        Background loop: continuously grab frames with auto-reconnect.
        Uses exponential backoff on connection failures.
        """
        delay = self.reconnect_delay

        while self._running:
            # Connect if not connected
            if not self._connected:
                # Mask password in logs
                safe_url = self.rtsp_url.split("@")[-1] if isinstance(self.rtsp_url, str) and "@" in self.rtsp_url else self.rtsp_url
                logger.info(f"Connecting to camera: {safe_url} ...")

                if self._connect():
                    delay = self.reconnect_delay  # Reset backoff
                else:
                    logger.warning(f"Reconnecting in {delay:.0f}s ...")
                    time.sleep(delay)
                    delay = min(delay * 2, self.max_reconnect_delay)
                    continue

            # Grab frames as fast as possible so stale packets are dropped quickly.
            ret, frame = self._cap.read()

            if not ret or frame is None:
                logger.warning("Frame grab failed — stream may have disconnected")
                self._connected = False
                continue

            # Update shared frame (thread-safe)
            with self._lock:
                self._frame = frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
