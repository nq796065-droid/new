import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path("D:/demo/skin-lesion-classification/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"[*] Starting download of HAM10000 dataset to {DATA_DIR}...")

# Download repository files excluding heavy CSVs that are redundant
snapshot_download(
    repo_id="aditya022/HAM10000_Dataset",
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    allow_patterns=[
        "archive/HAM10000_metadata.csv",
        "archive/HAM10000_images_part_1/*",
        "archive/HAM10000_images_part_2/*"
    ],
    max_workers=8
)

print("[*] Download completed. Verifying files...")

# Move or organize metadata
archive_dir = DATA_DIR / "archive"
metadata_src = archive_dir / "HAM10000_metadata.csv"
metadata_dst = DATA_DIR / "HAM10000_metadata.csv"

if metadata_src.exists() and not metadata_dst.exists():
    shutil.copy2(metadata_src, metadata_dst)
    print(f"[+] Metadata copied to {metadata_dst}")

# Create unified images directory
images_dir = DATA_DIR / "images"
images_dir.mkdir(exist_ok=True)

part1_dir = archive_dir / "HAM10000_images_part_1"
part2_dir = archive_dir / "HAM10000_images_part_2"

count = 0
for part_dir in [part1_dir, part2_dir]:
    if part_dir.exists():
        for img_file in part_dir.glob("*.jpg"):
            dest_file = images_dir / img_file.name
            if not dest_file.exists():
                shutil.move(img_file, dest_file)
            count += 1

print(f"[+] Unified {count} images in {images_dir}")
print(f"[+] Total images now in {images_dir}: {len(list(images_dir.glob('*.jpg')))}")
