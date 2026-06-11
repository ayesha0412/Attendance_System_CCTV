"""
Centroid Tracker
================
Associates face bounding boxes across video frames by centroid distance.
Returns stable track IDs so temporal voting works correctly.

Used by both ArcFace and AdaFace inference loops.
"""


class CentroidTracker:
    """Simple online tracker — matches detections to existing tracks by
    nearest-centroid within a pixel distance budget."""

    def __init__(self, max_dist=150, max_age=20):
        """
        Args:
            max_dist: Maximum pixel distance to associate a detection with a track.
            max_age:  Frames a track survives without a matching detection.
        """
        self._tracks  = {}     # {tid: {"cx", "cy", "age"}}
        self._next_id = 0
        self.max_dist = max_dist
        self.max_age  = max_age

    def update(self, bboxes):
        """
        Match new bboxes to existing tracks by centroid proximity.

        Args:
            bboxes: list of [x1, y1, x2, y2] in original frame coords.

        Returns:
            list of track_ids aligned 1:1 with the input bboxes.
        """
        # Age all existing tracks; drop stale ones
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self.max_age:
                del self._tracks[tid]

        if not bboxes:
            return []

        centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bboxes]
        assigned  = []
        used_tids = set()

        for cx, cy in centroids:
            best_tid, best_dist = None, self.max_dist

            for tid, t in self._tracks.items():
                if tid in used_tids:
                    continue
                dist = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is not None:
                self._tracks[best_tid].update({"cx": cx, "cy": cy, "age": 0})
                used_tids.add(best_tid)
            else:
                best_tid = self._next_id
                self._next_id += 1
                self._tracks[best_tid] = {"cx": cx, "cy": cy, "age": 0}

            assigned.append(best_tid)

        return assigned
