"""
Video Feed Overlay Drawing
===========================
Bounding-box rendering with corner brackets and confidence labels.
Shared by all recognition backends.
"""

import cv2

# ── Colors (BGR) ─────────────────────────────────────────────────────
GREEN = (0, 220, 0)
RED   = (0, 60, 220)
GRAY  = (140, 140, 140)
WHITE = (255, 255, 255)


def draw_box(frame, bbox, label, score, state):
    """
    Draw a styled bounding box with corner brackets and a label pill.

    Args:
        frame: BGR image to draw on (modified in-place).
        bbox:  [x1, y1, x2, y2] bounding box.
        label: Name string (or empty for identifying state).
        score: Float 0.0-1.0 confidence.
        state: 'known' | 'unknown' | 'identifying'
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = GREEN if state == "known" else (RED if state == "unknown" else GRAY)

    # Main rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner brackets — gives a modern HUD look
    cl = max(12, int((x2 - x1) * 0.15))
    t  = 3
    for (ax, ay), (bx, by) in [
        ((x1, y1), (x1 + cl, y1)), ((x1, y1), (x1, y1 + cl)),
        ((x2, y1), (x2 - cl, y1)), ((x2, y1), (x2, y1 + cl)),
        ((x1, y2), (x1 + cl, y2)), ((x1, y2), (x1, y2 - cl)),
        ((x2, y2), (x2 - cl, y2)), ((x2, y2), (x2, y2 - cl)),
    ]:
        cv2.line(frame, (ax, ay), (bx, by), color, t)

    # Label pill above the box
    text = "Identifying..." if state == "identifying" else f"{label}  {score:.0%}"
    font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
    ly1 = max(0, y1 - th - 12)
    ly2 = ly1 + th + 12
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 10, ly2), color, -1)
    cv2.putText(frame, text, (x1 + 5, ly2 - 4), font, fs, WHITE, ft, cv2.LINE_AA)
