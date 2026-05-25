import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ============ SETTINGS ============
_HERE        = os.path.dirname(os.path.abspath(__file__))

# Use the ORIGINAL clean crops for the gallery, not augmented ones.
# Augmented images (blur, dropout, perspective warp) produce noisy embeddings.
# Averaging 300+ distorted embeddings pulls the mean away from the person's
# real face — cosine similarity at inference then scores lower than expected.
# Clean originals give the tightest, most representative mean embedding.
DATA_DIR     = os.path.join(_HERE, "..", "Data_Augmentation", "Dataset_Clicked")
MODEL_OUTPUT = os.path.join(_HERE, "face_model.pkl")
DET_SCORE_MIN = 0.70   # skip low-confidence detections inside training images
# ==================================

# ============ STEP 1: Load ArcFace ============
print("[1/3] Loading ArcFace model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("      Model loaded!\n")

# ============ STEP 2: Extract embeddings ============
print("[2/3] Extracting face embeddings...")

person_embeddings = {}   # {person_name: [emb, emb, ...]}
skipped = 0

for person_name in sorted(os.listdir(DATA_DIR)):
    person_path = os.path.join(DATA_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    images = [f for f in os.listdir(person_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    embs = []
    for img_name in images:
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None:
            continue

        faces = app.get(img)
        if not faces:
            skipped += 1
            continue

        # Only keep high-confidence detections — blurry or side-profile
        # crops produce noisy embeddings that weaken the gallery mean.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if face.det_score < DET_SCORE_MIN:
            skipped += 1
            continue

        embs.append(face.embedding)

    if embs:
        person_embeddings[person_name] = embs
        print(f"  {person_name}: {len(embs)} embeddings")
    else:
        print(f"  {person_name}: 0 embeddings — skipped (no face detected in any image)")

print(f"\n  Total people: {len(person_embeddings)}")
print(f"  Skipped images (no face detected): {skipped}")

if not person_embeddings:
    print("\n[ERROR] No embeddings extracted! Check Data_Augmentation/Augmented_Data/")
    exit(1)

# ============ STEP 3: Build cosine-similarity gallery ============
# Each person gets one representative embedding = mean of all their
# L2-normalised embeddings, then re-normalised.
#
# Why not SVM?
#   ArcFace embeddings live on a unit hypersphere — the correct distance
#   metric is cosine similarity (dot product of normalised vectors).
#   SVM with an RBF kernel fights that geometry. Cosine gallery matching
#   is faster, more accurate, and doesn't need retraining when you add
#   a new employee — just extract their embeddings and insert.

print("\n[3/3] Building cosine-similarity gallery...")
gallery = {}

for name, embs in person_embeddings.items():
    arr = np.array(embs, dtype=np.float32)

    # L2-normalise every embedding so dot product == cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr   = arr / (norms + 1e-8)

    # Mean of normalised embeddings, then re-normalise
    mean_emb = arr.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    gallery[name] = mean_emb
    print(f"  {name}: gallery entry built from {len(embs)} embeddings")

print(f"\n  Gallery size: {len(gallery)} people")

with open(MODEL_OUTPUT, "wb") as f:
    pickle.dump({
        "gallery":     gallery,           # {name: L2-normalised mean embedding}
        "class_names": list(gallery.keys()),
    }, f)

print(f"\n[DONE] Gallery saved to {MODEL_OUTPUT}")
print(f"       Classes: {list(gallery.keys())}")
print(f"\n  Next step: python live_recognition.py")
