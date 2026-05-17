import cv2
import os
import shutil

# ============ SETTINGS ============
DATASET_DIR = "Data_Gathering"           # your main folder
THRESHOLD = 50                    # below this = blurry (adjust this!)
REJECTED_DIR = "dataset_rejected" # where bad images go
# ==================================

stats = {"total": 0, "kept": 0, "rejected": 0}

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    # Create rejected folder for this person
    rejected_person_path = os.path.join(REJECTED_DIR, person_name)
    os.makedirs(rejected_person_path, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # Skip non-image files
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot read: {img_path}")
            continue

        stats["total"] += 1

        # --- Laplacian blur score ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if score < THRESHOLD:
            # Move blurry image to rejected folder
            shutil.move(img_path, os.path.join(rejected_person_path, img_name))
            stats["rejected"] += 1
            print(f"[REJECTED] {img_path}  (score: {score:.1f})")
        else:
            stats["kept"] += 1
            print(f"[KEPT]     {img_path}  (score: {score:.1f})")

print("\n========== SUMMARY ==========")
print(f"Total images:    {stats['total']}")
print(f"Kept (sharp):    {stats['kept']}")
print(f"Rejected (blur): {stats['rejected']}")
print(f"Threshold used:  {THRESHOLD}")
print(f"\nRejected images moved to: {REJECTED_DIR}/")
print("\nTIP: If too many got rejected, lower THRESHOLD (e.g., 30)")
print("     If too few got rejected, raise THRESHOLD (e.g., 100)")