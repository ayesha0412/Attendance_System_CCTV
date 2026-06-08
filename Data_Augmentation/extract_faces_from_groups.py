"""
Extract individual face crops from group photos.

Input:  Data_Augmentation/Dataset_Clicked/Faces to Extract/   (group JPEGs)
Output: Data_Augmentation/Dataset_Clicked/extracted_faces/    (face crops)

Each output file is named:  <source_image_stem>_face_<N>.jpg
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent / "Dataset_Clicked" / "Faces to Extract"
OUTPUT_DIR = Path(__file__).parent / "Dataset_Clicked" / "extracted_faces"

CONFIDENCE   = 0.40   # lower than CCTV default — phone photos are higher quality
MIN_FACE_PX  = 40     # minimum face dimension to keep
PADDING_FRAC = 0.25   # pad bbox by 25% of face size on each side
# ─────────────────────────────────────────────────────────────────────────────


def load_detector() -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)
    return app


def pad_bbox(x1, y1, x2, y2, img_h, img_w, frac=PADDING_FRAC):
    fw, fh = x2 - x1, y2 - y1
    pad_x = int(fw * frac)
    pad_y = int(fh * frac)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    return x1, y1, x2, y2


def extract_faces(detector: FaceAnalysis):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.png"))
    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    total_saved = 0
    total_missed = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}")
            continue

        h, w = img.shape[:2]

        # Resize very large images for inference only (keep original for crop)
        infer = img
        scale = 1.0
        longest = max(h, w)
        if longest > 1920:
            scale = 1920 / longest
            infer = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = detector.get(infer)

        saved = 0
        for i, face in enumerate(faces, start=1):
            conf = float(face.det_score)
            if conf < CONFIDENCE:
                continue

            # Scale bbox back to original image coords
            x1, y1, x2, y2 = face.bbox.astype(np.float32)
            if scale != 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            fw, fh = x2 - x1, y2 - y1
            if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                continue

            # Pad and crop from original image
            px1, py1, px2, py2 = pad_bbox(x1, y1, x2, y2, h, w)
            crop = img[py1:py2, px1:px2]

            out_name = f"{img_path.stem}_face_{i:03d}.jpg"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        status = f"  {saved} face(s) saved" if saved else "  [!] no faces detected"
        print(f"{img_path.name}: {status}")

        total_saved += saved
        if saved == 0:
            total_missed += 1

    print(f"\nDone. {total_saved} face crops saved to: {OUTPUT_DIR}")
    if total_missed:
        print(f"       {total_missed} image(s) yielded no detections — check manually.")


if __name__ == "__main__":
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")
    detector = load_detector()
    extract_faces(detector)
