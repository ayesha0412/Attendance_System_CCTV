"""
Frame Pre-processing
====================
CLAHE (Contrast-Limited Adaptive Histogram Equalisation) on the L channel
to normalise uneven CCTV / IR lighting before embedding extraction.

Recommended for ArcFace which benefits from consistent lighting.
AdaFace handles low-quality images internally so CLAHE is optional there.
"""

import cv2

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_clahe(frame):
    """Apply CLAHE to the lightness channel of a BGR frame."""
    lab    = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l      = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
