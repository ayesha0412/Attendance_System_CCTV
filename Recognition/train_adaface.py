"""
AdaFace Gallery Builder
========================
Detects faces with InsightFace SCRFD (same as always),
aligns them to 112x112, extracts embeddings with AdaFace IR-50,
then builds a cosine-similarity gallery and saves it.

AdaFace is specifically designed for low-quality / surveillance images.
It adapts the training margin based on image quality, giving much better
recall on blurry, angled, or partially-occluded CCTV faces vs ArcFace.

Output: Recognition/face_model_adaface.pkl
Usage:  python train_adaface.py
"""

import os
import sys
import cv2
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis

# ============ SETTINGS ============
_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(_HERE, "adaface_assets", "cvlface_model")

# Use original clean crops for the gallery — not augmented ones.
# (Augmented embeddings are noisy; a clean mean is more discriminative.)
DATA_DIR     = os.path.join(_HERE, "..", "Data_Augmentation", "Augmented_Data")
MODEL_OUTPUT = os.path.join(_HERE, "face_model_adaface.pkl")

DET_SCORE_MIN = 0.70   # skip low-confidence detections in source images
# ==================================


# =====================================================================
#  Face alignment — standard 112x112 used by ArcFace AND AdaFace
# =====================================================================
# Reference landmark positions in the 112x112 output image
_LANDMARK_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_112(img, kps):
    """
    Warp img so that the 5 detected landmarks (kps, shape 5x2)
    land on the canonical 112x112 positions used by ArcFace/AdaFace.
    Uses OpenCV estimateAffinePartial2D (similarity transform).
    """
    M, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32).reshape(-1, 1, 2),
        _LANDMARK_DST.reshape(-1, 1, 2),
    )
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


# =====================================================================
#  AdaFace model loader
# =====================================================================
def load_adaface(model_dir):
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"[ERROR] Model not found at {model_dir}")
        print("        Run: python setup_adaface.py")
        sys.exit(1)

    # Load directly from local model code — bypasses transformers version mismatches.
    # The cvlface_model uses relative paths and a local 'models' package, so we must
    # chdir into the model directory and prepend it to sys.path first.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from models.iresnet import load_model as _load_iresnet
        from omegaconf import OmegaConf
        import yaml
        conf = OmegaConf.create(yaml.safe_load(open("pretrained_model/model.yaml")))
        model = _load_iresnet(conf)
        model.load_state_dict_from_path("pretrained_model/model.pt")
    finally:
        os.chdir(_prev_cwd)

    model.eval()

    # RTX 5080 (Blackwell SM 12.0) needs PyTorch nightly for CUDA kernel support.
    # Run a tiny forward pass on CUDA to confirm compatibility; fall back to CPU otherwise.
    device = "cpu"
    if torch.cuda.is_available():
        try:
            model.cuda()
            with torch.no_grad():
                model(torch.zeros(1, 3, 112, 112).cuda())
            device = "cuda"
        except Exception:
            model.cpu()
            device = "cpu"
            print("[WARN] CUDA not supported for this GPU by current PyTorch — AdaFace on CPU.")

    print(f"[OK] AdaFace IR-50 loaded on {device.upper()}")
    return model, device


def get_embedding(model, device, aligned_bgr):
    """Convert aligned BGR 112x112 → AdaFace embedding (512-dim, L2-normalised)."""
    img    = aligned_bgr[:, :, ::-1].astype(np.float32)   # BGR → RGB
    img    = (img / 255.0 - 0.5) / 0.5                    # normalise to [-1, 1]
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)   # IResNetModel.forward(x) — plain tensor
    if isinstance(output, (list, tuple)):
        emb = output[0]
    else:
        emb = output
    if emb.dim() > 2:
        emb = emb[:, 0]
    emb = F.normalize(emb, dim=1)
    return emb.squeeze().cpu().numpy()


# =====================================================================
#  Helpers
# =====================================================================
def _padded_candidates(img):
    """Yield the raw crop, then a padded version (50% border) for SCRFD."""
    yield img
    h, w = img.shape[:2]
    pad = max(h, w) // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    yield padded


# =====================================================================
#  Main
# =====================================================================
def main():
    print("=" * 55)
    print("  AdaFace Gallery Builder")
    print("=" * 55)
    print()

    # Load AdaFace
    print("[1/3] Loading AdaFace IR-50...")
    adaface_model, device = load_adaface(MODEL_DIR)
    print()

    # Load InsightFace SCRFD for detection + landmarks only
    print("[..] Loading InsightFace SCRFD (detection only)...")
    detector = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    print("[OK] Detector ready.\n")

    # Extract embeddings
    print("[2/3] Extracting AdaFace embeddings from original crops...")
    person_embeddings = {}

    for person_name in sorted(os.listdir(DATA_DIR)):
        person_path = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

        embs = []
        fallback_count = 0
        for img_name in images:
            img = cv2.imread(os.path.join(person_path, img_name))
            if img is None:
                continue

            aligned = None

            # The source images are already face crops, so SCRFD often misses them
            # (no background context). Try twice: once on the raw crop, once with
            # padding added so the detector has context to anchor landmarks.
            for attempt_img in _padded_candidates(img):
                faces = detector.get(attempt_img)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                if face.det_score >= DET_SCORE_MIN and face.kps is not None:
                    aligned = align_112(attempt_img, face.kps)
                    if aligned is not None:
                        break

            if aligned is None:
                # Last resort: direct resize. Valid because images are already face crops.
                aligned = cv2.resize(img, (112, 112))
                fallback_count += 1

            emb = get_embedding(adaface_model, device, aligned)
            embs.append(emb)

        if fallback_count:
            print(f"  {person_name}: {len(embs)} embeddings "
                  f"({len(embs) - fallback_count} aligned, {fallback_count} resized)")

        if embs:
            person_embeddings[person_name] = embs
            if not fallback_count:
                print(f"  {person_name}: {len(embs)} embeddings (all aligned)")
        else:
            print(f"  {person_name}: 0 embeddings — skipped")

    print(f"\n  Total people: {len(person_embeddings)}")

    if not person_embeddings:
        print("\n[ERROR] No embeddings extracted. Check Data_Gathering has images.")
        sys.exit(1)

    # Build gallery
    print("\n[3/3] Building cosine-similarity gallery...")
    gallery = {}
    for name, embs in person_embeddings.items():
        arr  = np.array(embs, dtype=np.float32)
        arr  = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        mean = arr.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        gallery[name] = mean
        print(f"  {name}: gallery entry from {len(embs)} embeddings")

    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump({
            "gallery":     gallery,
            "class_names": list(gallery.keys()),
            "model":       "adaface_ir50_ms1mv3",
        }, f)

    print(f"\n[DONE] AdaFace gallery saved → {MODEL_OUTPUT}")
    print(f"       Classes: {list(gallery.keys())}")
    print(f"\n  Next: python live_adaface.py")


if __name__ == "__main__":
    main()
