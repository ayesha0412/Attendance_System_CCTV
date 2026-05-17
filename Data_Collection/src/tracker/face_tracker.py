"""Face tracker with ByteTrack primary backend and IoU fallback."""

import time
import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import supervision as sv

logger = logging.getLogger("face_collector.tracker")


@dataclass
class Track:
    """A tracked face identity across frames."""
    track_id: int
    bbox: np.ndarray                # Latest bounding box [x1, y1, x2, y2]
    last_seen: float                # Timestamp of last detection
    save_count: int = 0             # Number of times this face has been saved
    last_saved: float = 0.0         # Timestamp of last save
    frames_since_seen: int = 0     # Reserved for debug/telemetry


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


class FaceTracker:
    """Track faces and emit saveable detections at fixed time intervals."""

    def __init__(self, iou_threshold: float = 0.4,
                 cooldown_seconds: float = 2.0,
                 max_faces: int = 20,
                 stale_timeout: int = 30,
                 backend: str = "bytetrack"):
        self.iou_threshold = iou_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_faces = max_faces
        self.stale_timeout = stale_timeout
        self.backend = backend.lower()

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1
        self._total_saved = 0

        self._tracker = None
        if self.backend == "bytetrack":
            # The frame-rate is only used to calibrate track age logic inside ByteTrack.
            self._tracker = sv.ByteTrack(frame_rate=10)
            logger.info("Tracker backend: ByteTrack")
        else:
            logger.info("Tracker backend: IoU fallback")

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    @property
    def total_saved(self) -> int:
        return self._total_saved

    def update(self, detections: list) -> list:
        """
        Match new detections to existing tracks.

        Args:
            detections: List of DetectedFace objects with .bbox attribute

        Returns:
            List of (track_id, DetectedFace) tuples for faces that should be saved
            (either new faces or faces whose cooldown has expired).
        """
        now = time.time()
        faces_to_save = []
        matched_track_ids = set()

        if self._tracker is not None:
            return self._update_bytetrack(detections)

        # Sort detections by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda f: f.confidence, reverse=True)

        for face in sorted_dets:
            best_iou = 0.0
            best_track_id = None

            # Find best matching track
            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = compute_iou(face.bbox, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = tid

            if best_iou >= self.iou_threshold and best_track_id is not None:
                # Matched to existing track — update it
                track = self._tracks[best_track_id]
                track.bbox = face.bbox
                track.last_seen = now
                matched_track_ids.add(best_track_id)

                # Check cooldown — should we save again?
                if (now - track.last_saved) >= self.cooldown_seconds:
                    track.last_saved = now
                    track.save_count += 1
                    self._total_saved += 1
                    faces_to_save.append((best_track_id, face))
            else:
                # New face — create track
                if len(self._tracks) < self.max_faces:
                    track_id = self._next_id
                    self._next_id += 1
                    self._tracks[track_id] = Track(
                        track_id=track_id,
                        bbox=face.bbox,
                        last_seen=now,
                        last_saved=now,
                        save_count=1,
                    )
                    self._total_saved += 1
                    faces_to_save.append((track_id, face))
                    logger.debug(f"New track #{track_id} created")

        # Prune stale tracks
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if (now - track.last_seen) > self.stale_timeout
        ]
        for tid in stale_ids:
            del self._tracks[tid]
            logger.debug(f"Track #{tid} pruned (stale)")

        return faces_to_save

    def _update_bytetrack(self, detections: list) -> list:
        """Use ByteTrack IDs and save every cooldown window per track."""
        now = time.time()
        faces_to_save = []

        if not detections:
            return faces_to_save

        xyxy = np.array([face.bbox for face in detections], dtype=np.float32)
        confidence = np.array([face.confidence for face in detections], dtype=np.float32)
        class_id = np.zeros(len(detections), dtype=np.int32)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_dets)

        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else np.array([], dtype=np.int32)
        used_det_indices = set()
        for i in range(len(tracked)):
            track_id = int(tracker_ids[i])
            track_box = tracked.xyxy[i]

            # Associate tracked box to current detections using IoU.
            best_idx = None
            best_iou = 0.0
            for det_idx, det_face in enumerate(detections):
                if det_idx in used_det_indices:
                    continue
                iou = compute_iou(track_box, det_face.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = det_idx

            if best_idx is None or best_iou < 0.1:
                continue

            used_det_indices.add(best_idx)
            face = detections[best_idx]
            face.bbox = track_box.astype(int)

            track = self._tracks.get(track_id)
            if track is None:
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=face.bbox,
                    last_seen=now,
                    last_saved=now,
                    save_count=1,
                    frames_since_seen=0,
                )
                self._total_saved += 1
                faces_to_save.append((track_id, face))
                continue

            track.bbox = face.bbox
            track.last_seen = now
            track.frames_since_seen = 0

            if (now - track.last_saved) >= self.cooldown_seconds:
                track.last_saved = now
                track.save_count += 1
                self._total_saved += 1
                faces_to_save.append((track_id, face))

        # Prune IDs that have not been refreshed recently.
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if (now - track.last_seen) > self.stale_timeout
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        return faces_to_save

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
        logger.info("Tracker reset")
