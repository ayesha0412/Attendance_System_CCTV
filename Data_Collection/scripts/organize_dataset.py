"""
Dataset Organizer
-----------------
After collecting raw face images, use this tool to:
1. Browse collected faces
2. Manually assign them to named person folders
3. Auto-cluster similar faces using embeddings (optional, requires InsightFace)

Usage:
  python scripts/organize_dataset.py --input data/raw_faces --output data/dataset
"""

import cv2
import os
import shutil
import argparse
from pathlib import Path


def manual_organizer(input_dir: str, output_dir: str):
    """
    Simple interactive organizer.
    Shows each face and asks for a folder name (person ID).
    Type 's' to skip, 'q' to quit.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
    print(f"Found {len(images)} images in {input_dir}\n")

    skipped = 0
    assigned = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Show the face
        display = cv2.resize(img, (200, 200))
        cv2.imshow("Face - Enter person ID (s=skip, q=quit)", display)
        cv2.waitKey(100)

        person_id = input(f"  {img_path.name} → Person ID (or 's' to skip): ").strip()

        if person_id.lower() == 'q':
            print("Quitting.")
            break
        elif person_id.lower() == 's' or not person_id:
            skipped += 1
            continue

        # Move to person folder
        person_folder = output_path / person_id
        person_folder.mkdir(exist_ok=True)
        dest = person_folder / img_path.name
        shutil.copy2(str(img_path), str(dest))
        print(f"  → Saved to {dest}")
        assigned += 1

    cv2.destroyAllWindows()
    print(f"\nDone. Assigned: {assigned}, Skipped: {skipped}")


def print_dataset_stats(dataset_dir: str):
    """Print summary of organized dataset."""
    path = Path(dataset_dir)
    if not path.exists():
        print(f"Directory {dataset_dir} does not exist.")
        return

    persons = [d for d in path.iterdir() if d.is_dir()]
    print(f"\n=== Dataset Stats: {dataset_dir} ===")
    print(f"Total persons: {len(persons)}")
    for p in sorted(persons):
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png"))
        print(f"  {p.name}: {len(imgs)} images")
    total = sum(len(list(p.glob("*.jpg")) + list(p.glob("*.png"))) for p in persons)
    print(f"Total images: {total}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw_faces", help="Raw collected faces directory")
    parser.add_argument("--output", default="data/dataset", help="Organized dataset directory")
    parser.add_argument("--stats", action="store_true", help="Only print dataset stats")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.output)
    else:
        manual_organizer(args.input, args.output)
        print_dataset_stats(args.output)