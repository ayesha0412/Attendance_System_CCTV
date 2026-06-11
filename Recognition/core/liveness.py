"""
Anti-Spoofing / Liveness Detection
===================================
Rejects phone screens, printed photos, and tablet video replays using
three fast image-level heuristics.  No extra model required.

Heuristics:
  1. Frequency spectrum — screens produce periodic moire patterns
  2. Saturation variance — real skin varies; screens have narrow gamut
  3. Laplacian variance — real faces have depth-driven edges; flat media don't

All three must fail simultaneously for a rejection (logical AND),
which keeps the false-positive rate very low on real faces.
"""

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


def check_liveness(face_bgr):
    """
    Return True if the face appears to be a real person.
    Return False if the face looks like a screen or printed photo.

    Args:
        face_bgr: BGR image of the face (any size — aligned 112x112 or bbox crop).
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Frequency ratio — high-frequency energy vs total
    f         = np.fft.fft2(gray.astype(np.float32))
    magnitude = np.log1p(np.abs(np.fft.fftshift(f)))
    h, w      = magnitude.shape
    ch, cw    = h // 2, w // 2
    high_freq = magnitude.copy()
    high_freq[ch - 10:ch + 10, cw - 10:cw + 10] = 0
    freq_ratio = np.sum(high_freq) / (np.sum(magnitude) + 1e-8)

    # 2. Saturation standard deviation
    hsv     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))

    # 3. Laplacian variance (edge sharpness)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_fake = freq_ratio > 0.85 and sat_std < 25 and lap_var < 100

    if is_fake:
        logger.warning("Spoof rejected: freq=%.2f  sat_std=%.1f  lap=%.1f",
                       freq_ratio, sat_std, lap_var)

    return not is_fake
