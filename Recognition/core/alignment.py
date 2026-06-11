"""
Face Alignment
==============
Warps a detected face using 5 facial landmarks to the canonical 112x112
positions used by ArcFace and AdaFace embedding models.

The similarity transform (rotation + uniform scale + translation) ensures
that eye positions, nose, and mouth corners land on fixed pixel coordinates,
making embeddings invariant to head pose and camera angle.
"""

import cv2
import numpy as np

# Canonical 5-point landmark positions in 112x112 output image.
# These are the standard positions used by InsightFace / ArcFace / AdaFace.
LANDMARK_DST = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_112(img, kps):
    """
    Align a face to 112x112 using a similarity transform from 5 landmarks.

    Args:
        img: Full BGR frame (any size).
        kps: 5x2 numpy array of detected landmark coordinates.

    Returns:
        112x112 aligned BGR image, or None if the transform fails.
    """
    M, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32).reshape(-1, 1, 2),
        LANDMARK_DST.reshape(-1, 1, 2),
    )
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
