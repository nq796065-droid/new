"""
data.py
Dataset handling, Patient-aware Group Stratification, Augmentations, and Samplers
for Skin-Lesion Classification (HAM10000).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold


# Standard class mapping
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_FULL_NAMES = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(aug_mode: str = "basic", img_size: int = 224) -> transforms.Compose:
    """
    Returns image transformation pipeline based on augmentation mode.
    Modes:
      - 'basic': standard horizontal/vertical flip
      - 'advanced': flip, slight rotation, color jitter, affine translate
      - 'none' or 'eval': deterministic resizing and normalization
    """
    if aug_mode == "basic":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    elif aug_mode in ["advanced", "heavy"]:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:  # eval / test / val
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesions.
    """
    def __init__(self, df: pd.DataFrame, img_dir_map: Dict[str, Path], transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir_map = img_dir_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = CLASS_TO_IDX[row['dx']]

        img_path = self.img_dir_map.get(image_id)
        if img_path is None or not os.path.exists(img_path):
            raise FileNotFoundError(f"Image {image_id} not found in search paths.")

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_id


def build_image_map(search_dirs: List[Path]) -> Dict[str, Path]:
    """
    Maps image_id -> full Path for fast O(1) loading.
    """
    img_map = {}
    for d in search_dirs:
        if d.exists():
            for p in d.rglob("*.jpg"):
                img_id = p.stem
                if img_id not in img_map:
                    img_map[img_id] = p
    return img_map


def prepare_data_splits(metadata_csv: Path, splits_csv: Path, seed: int = 42) -> pd.DataFrame:
    """
    Splits dataset into 70% train, 15% val, 15% test.
    CRITICAL: Grouped by lesion_id using StratifiedGroupKFold to prevent data leakage!
    """
    df = pd.read_csv(metadata_csv)
    if 'dx' not in df.columns or 'lesion_id' not in df.columns:
        raise ValueError("Metadata CSV must contain 'dx' and 'lesion_id' columns.")

    if splits_csv.exists():
        print(f"[*] Loading existing split from {splits_csv}")
        return pd.read_csv(splits_csv)

    print(f"[*] Creating leakage-free Stratified Group split (Seed {seed})...")
    # Step 1: Create 10 folds using StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (_, test_indices) in enumerate(sgkf.split(df, df['dx'], df['lesion_id'])):
        folds[test_indices] = fold_idx

    # Assign: folds 0..6 (70%) -> train, folds 7,8 (~15%) -> val, fold 9 (~15%) -> test
    # To get ~70/15/15: folds 0..6 (70%), folds 7,8 (20% -> let's balance fold sizes)
    # Let's do a 2-stage split: 85% train_val, 15% test (sgkf with 7 folds -> 1 fold is ~14.3%)
    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(df, df['dx'], df['lesion_id']))

    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True

    df_train_val = df[~test_mask].reset_index()
    # Next split train_val into train (82.3% of train_val = 70% of total) and val (17.7% of train_val = 15% of total)
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_sub_idx, val_sub_idx = next(sgkf_val.split(df_train_val, df_train_val['dx'], df_train_val['lesion_id']))

    split_col = np.array(['train'] * len(df), dtype=object)
    split_col[test_idx] = 'test'
    original_val_idx = df_train_val.iloc[val_sub_idx]['index'].values
    split_col[original_val_idx] = 'val'

    df['split'] = split_col
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_csv, index=False)
    print(f"[+] Splits saved successfully to {splits_csv}")

    # Verify no lesion_id leakage
    train_lesions = set(df[df['split'] == 'train']['lesion_id'])
    val_lesions = set(df[df['split'] == 'val']['lesion_id'])
    test_lesions = set(df[df['split'] == 'test']['lesion_id'])

    assert len(train_lesions.intersection(val_lesions)) == 0, "DATA LEAKAGE DETECTED BETWEEN TRAIN AND VAL!"
    assert len(train_lesions.intersection(test_lesions)) == 0, "DATA LEAKAGE DETECTED BETWEEN TRAIN AND TEST!"
    assert len(val_lesions.intersection(test_lesions)) == 0, "DATA LEAKAGE DETECTED BETWEEN VAL AND TEST!"
    print("[+] Verified ZERO lesion_id overlap across splits (No Data Leakage!).")

    return df


def get_data_loaders(
    data_dir: Path,
    batch_size: int = 32,
    aug_mode: str = "basic",
    sampling_mode: str = "random",
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[int], pd.DataFrame]:
    """
    Prepares DataLoaders for train, val, and test splits.
    sampling_mode options:
      - 'random': Default random shuffle
      - 'weighted': WeightedRandomSampler inverse class weighting
      - 'balanced': Sample equally across classes
    """
    # Locate metadata
    possible_meta = [
        data_dir / "HAM10000_metadata.csv",
        data_dir / "archive" / "HAM10000_metadata.csv"
    ]
    meta_path = next((p for p in possible_meta if p.exists()), None)
    if meta_path is None:
        raise FileNotFoundError(f"Could not find HAM10000_metadata.csv in {data_dir}")

    splits_csv = data_dir / "splits.csv"
    df = prepare_data_splits(meta_path, splits_csv, seed=seed)

    # Search folders for images
    search_dirs = [
        data_dir / "images",
        data_dir / "archive" / "HAM10000_images_part_1",
        data_dir / "archive" / "HAM10000_images_part_2",
        data_dir / "HAM10000_images_part_1",
        data_dir / "HAM10000_images_part_2",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "kmader" / "skin-cancer-mnist-ham10000"
    ]
    img_map = build_image_map(search_dirs)

    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Calculate class counts in training set
    train_class_counts = [int((train_df['dx'] == cls_name).sum()) for cls_name in CLASS_NAMES]

    train_transform = get_transforms(aug_mode=aug_mode)
    eval_transform = get_transforms(aug_mode="eval")

    train_dataset = HAM10000Dataset(train_df, img_map, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, img_map, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, img_map, transform=eval_transform)

    # Configure Sampler
    sampler = None
    shuffle = True
    if sampling_mode == "weighted":
        class_weights = 1.0 / (np.array(train_class_counts, dtype=np.float32) + 1e-5)
        train_labels = [CLASS_TO_IDX[dx] for dx in train_df['dx']]
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
        shuffle = False
    elif sampling_mode == "balanced":
        # Weight inversely to make every class equal probability in sampling
        class_weights = 1.0 / np.array(train_class_counts, dtype=np.float32)
        class_weights /= class_weights.sum()
        train_labels = [CLASS_TO_IDX[dx] for dx in train_df['dx']]
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader, train_class_counts, df
