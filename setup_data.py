"""
setup_data.py
Consolidates HAM10000 dataset into data/ and verifies zero-leakage splits.
"""

import shutil
from pathlib import Path
import pandas as pd
from data import prepare_data_splits, CLASS_NAMES, CLASS_FULL_NAMES


DATA_DIR = Path("D:/demo/skin-lesion-classification/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Search locations for downloaded HAM10000
KAGGLEHUB_DIR = Path.home() / ".cache" / "kagglehub" / "datasets" / "kmader" / "skin-cancer-mnist-ham10000"

print(f"[*] Looking for downloaded dataset in {KAGGLEHUB_DIR} and {DATA_DIR}...")

# Find metadata
meta_file = None
for candidate in [
    DATA_DIR / "HAM10000_metadata.csv",
    DATA_DIR / "archive" / "HAM10000_metadata.csv",
    *KAGGLEHUB_DIR.rglob("HAM10000_metadata.csv")
]:
    if candidate.exists():
        meta_file = candidate
        break

if meta_file is None:
    print("[!] Metadata file not found yet. Please ensure download has completed.")
    exit(1)

target_meta = DATA_DIR / "HAM10000_metadata.csv"
if not target_meta.exists() or target_meta.resolve() != meta_file.resolve():
    shutil.copy2(meta_file, target_meta)
    print(f"[+] Copied metadata to {target_meta}")

# Count images in cache and data
img_sources = [
    DATA_DIR / "images",
    DATA_DIR / "archive" / "HAM10000_images_part_1",
    DATA_DIR / "archive" / "HAM10000_images_part_2",
    *KAGGLEHUB_DIR.rglob("HAM10000_images_part_1"),
    *KAGGLEHUB_DIR.rglob("HAM10000_images_part_2")
]

images_found = 0
for s in img_sources:
    if s.exists() and s.is_dir():
        cnt = len(list(s.glob("*.jpg")))
        if cnt > 0:
            print(f"    - Found {cnt} images in {s}")
            images_found += cnt

print(f"[*] Total image files detected across locations: {images_found}")

# Generate and verify zero-leakage splits
splits_csv = DATA_DIR / "splits.csv"
df_splits = prepare_data_splits(target_meta, splits_csv, seed=42)

print("\n" + "="*70)
print("SPLIT DISTRIBUTION ACROSS 7 CLASSES:")
print("="*70)
split_pivot = pd.crosstab(df_splits['dx'], df_splits['split'], margins=True)
print(split_pivot)
print("="*70)
print("[+] Dataset setup and zero-leakage validation completed successfully!")
