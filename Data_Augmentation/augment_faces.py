import cv2
import os
import albumentations as A
import numpy as np

# ============ SETTINGS ============
INPUT_DIR = "Data_Gathering"
OUTPUT_DIR = "Augmented_Data"
AUGMENTATIONS_PER_IMAGE = 40   # base count; auto-scales up for small datasets
MIN_AUGMENTED_PER_PERSON = 300 # if a person has fewer augmented images than
                                # this, multiplier is raised automatically
# ==================================

# Augmentation pipeline — designed for CCTV face crops:
#
# Tier 1 (geometry): simulates camera angle, head pose, mount position
# Tier 2 (photometric): covers day/night shift, IR lighting, compression
# Tier 3 (occlusion): masks, hats, hands, partial exit from frame
transform = A.Compose([
    # --- Geometry ---
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT, p=0.6),
    A.RandomScale(scale_limit=0.2, p=0.4),
    # Perspective warp: critical for CCTV where camera is wall/ceiling-mounted
    A.Perspective(scale=(0.02, 0.08), keep_size=True, p=0.4),

    # --- Photometric ---
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=0.5),
    A.CLAHE(clip_limit=3.0, p=0.3),
    # Sharpen: many CCTV encoders apply digital sharpening
    A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.25),
    # Simulates CCTV H.264/H.265 block artifacts and low-bitrate compression
    A.ImageCompression(quality_lower=30, quality_upper=85, p=0.35),

    # --- Noise / blur ---
    A.GaussNoise(var_limit=(10, 60), p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),

    # --- Occlusion ---
    # Simulates glasses, mask edge, hand, cap brim across the face
    A.CoarseDropout(
        max_holes=5, max_height=20, max_width=20,
        min_holes=1, min_height=5, min_width=5,
        fill_value=0, p=0.35,
    ),
    # Shadow stripe: lamp / door-frame shadow falling across the face
    A.RandomShadow(
        shadow_roi=(0, 0.0, 1, 1),
        num_shadows_lower=1, num_shadows_upper=2,
        shadow_dimension=4,
        p=0.25,
    ),
])

stats = {"total_original": 0, "total_augmented": 0}

for person_name in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    # Create output folder for this person
    output_person_path = os.path.join(OUTPUT_DIR, person_name)
    os.makedirs(output_person_path, exist_ok=True)

    images = [f for f in os.listdir(person_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    n_orig = len(images)
    if n_orig == 0:
        print(f"\n[{person_name}] No images — skipping")
        continue

    # Auto-scale multiplier so every person gets at least MIN_AUGMENTED_PER_PERSON
    # augmented images regardless of how few originals they have.
    aug_count = max(AUGMENTATIONS_PER_IMAGE,
                    int(np.ceil(MIN_AUGMENTED_PER_PERSON / n_orig)))
    print(f"\n[{person_name}] {n_orig} originals → {aug_count}x augmentation "
          f"= {n_orig * (aug_count + 1)} total images")

    for img_name in images:
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        stats["total_original"] += 1
        base_name = os.path.splitext(img_name)[0]

        # Copy original to output
        cv2.imwrite(os.path.join(output_person_path, f"{base_name}_original.png"), img)

        # Generate augmented versions
        for i in range(aug_count):
            augmented = transform(image=img)["image"]
            out_name = f"{base_name}_aug_{i+1:03d}.png"
            cv2.imwrite(os.path.join(output_person_path, out_name), augmented)

        stats["total_augmented"] += aug_count + 1  # +1 for original

print("\n========== SUMMARY ==========")
print(f"Original images:  {stats['total_original']}")
print(f"Total output:     {stats['total_augmented']}")
print(f"Base multiplier:  {AUGMENTATIONS_PER_IMAGE}x (auto-scaled per person)")
print(f"Output folder:    {OUTPUT_DIR}/")
print(f"\nNext step: run train_embeddings.py to rebuild face_model.pkl")