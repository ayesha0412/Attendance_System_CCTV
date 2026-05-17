"""Face detector wrapper around InsightFace SCRFD."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger("face_collector.detector")


@dataclass
class DetectedFace:
    """Single detected face with metadata."""
    bbox: np.ndarray            # [x1, y1, x2, y2]
    confidence: float           # Detection confidence 0-1
    landmarks: np.ndarray       # 5-point facial landmarks (shape: 5×2)
    embedding: Optional[np.ndarray] = None  # Reserved for future use

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple:
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        return (cx, cy)


class FaceDetector:
    """Face detector using InsightFace SCRFD through FaceAnalysis."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_name: str = "buffalo_l",
                 confidence: float = 0.5,
                 min_face_size: int = 80,
                 gpu_id: int = 0,
                 input_size: tuple = (640, 640),
                 max_frame_side: int = 1280):
        self.model_path = model_path
        self.model_name = model_name
        self.confidence = confidence
        self.min_face_size = min_face_size
        self.input_size = input_size
        self.max_frame_side = max_frame_side
        self.gpu_id = gpu_id
        self._detector = None

    def initialize(self):
        """Load SCRFD via FaceAnalysis and prefer GPU execution."""
        # Use a permissive detector threshold internally, then apply project-level
        # confidence filtering in detect() for better CCTV recall.
        internal_det_thresh = 0.1

        app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        try:
            app.prepare(ctx_id=self.gpu_id, det_size=tuple(self.input_size), det_thresh=internal_det_thresh)
            det_model = app.models.get("detection") if hasattr(app, "models") else None
            providers = []
            if det_model is not None and hasattr(det_model, "session"):
                providers = det_model.session.get_providers()

            if "CUDAExecutionProvider" in providers:
                logger.info("SCRFD initialized on GPU (CUDAExecutionProvider)")
            else:
                logger.warning("SCRFD running on CPU providers: %s", providers)
        except Exception as exc:
            logger.warning("GPU initialization failed (%s). Falling back to CPU.", exc)
            app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=tuple(self.input_size), det_thresh=internal_det_thresh)
            logger.info("SCRFD initialized on CPU")

        self._detector = app

    def detect(self, frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a BGR frame.

        Returns a list of DetectedFace filtered by confidence and min size.
        """
        if self._detector is None:
            raise RuntimeError("Detector not initialized. Call initialize() first.")

        h, w = frame.shape[:2]
        scale = 1.0
        infer_frame = frame
        longest = max(h, w)
        if longest > self.max_frame_side:
            scale = self.max_frame_side / float(longest)
            infer_w = max(1, int(w * scale))
            infer_h = max(1, int(h * scale))
            infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)

        faces = self._detector.get(infer_frame)

        # Max face size ratio — reject detections that cover > 40% of the frame
        # (those are false positives like doors/walls, not real faces)
        max_face_w = int(w * scale * 0.4)
        max_face_h = int(h * scale * 0.4)

        results = []

        for face in faces:
            conf = float(face.det_score)
            if conf < self.confidence:
                continue

            raw_bbox = face.bbox.astype(np.float32)
            raw_w = int(raw_bbox[2] - raw_bbox[0])
            raw_h = int(raw_bbox[3] - raw_bbox[1])

            # Skip absurdly large detections (false positives)
            if raw_w > max_face_w or raw_h > max_face_h:
                logger.debug(f"Skipped oversized detection: {raw_w}x{raw_h}px (max={max_face_w}x{max_face_h})")
                continue

            x1, y1, x2, y2 = raw_bbox
            if scale != 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            fw = int(x2 - x1)
            fh = int(y2 - y1)

            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            landmarks = np.asarray(face.kps, dtype=np.float32) if face.kps is not None else np.zeros((5, 2), dtype=np.float32)
            if scale != 1.0 and landmarks.size > 0:
                landmarks = landmarks / scale

            results.append(DetectedFace(
                bbox=np.array([x1, y1, x2, y2]),
                confidence=conf,
                landmarks=landmarks,
            ))

        if results:
            confs = [f"{f.confidence:.2f}" for f in results]
            sizes = [f"{f.width}x{f.height}" for f in results]
            logger.info(f"Detected {len(results)} faces — conf={confs}, sizes={sizes} (frame={w}x{h})")

        return results

    def draw_detections(self, frame: np.ndarray, faces: List[DetectedFace],
                        color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw highly visible bounding boxes on frame for dashboard preview.
        Uses an expanded padding area so tiny faces are easier to spot.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            fw = x2 - x1
            fh = y2 - y1

            # Draw an expanded highlight area (2x padding) so small faces are visible
            pad = max(fw, fh, 30)  # At least 30px padding
            ex1 = max(0, x1 - pad)
            ey1 = max(0, y1 - pad)
            ex2 = min(w, x2 + pad)
            ey2 = min(h, y2 + pad)

            # Semi-transparent outer rectangle (highlight area)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # Solid bright green inner rectangle (actual face bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Confidence label with background
            label = f"FACE {face.confidence:.0%}"
            font_scale = 0.6
            label_thickness = 2
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)[0]
            label_y = max(ey1, 20)  # Don't go above frame
            cv2.rectangle(annotated,
                          (ex1, label_y - label_size[1] - 8),
                          (ex1 + label_size[0] + 8, label_y),
                          (0, 255, 0), -1)
            cv2.putText(annotated, label, (ex1 + 4, label_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), label_thickness)

        return annotated
