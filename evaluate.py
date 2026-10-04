"""
evaluate.py
Comprehensive evaluation module for Skin Lesion Classification.
Calculates:
  - Overall Accuracy
  - Macro F1-score (Key Metric)
  - Weighted F1-score
  - Balanced Accuracy (Macro Recall)
  - Per-class Precision, Recall, and F1-score
  - Normalized Confusion Matrix plot
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import DataLoader

from data import CLASS_NAMES, CLASS_FULL_NAMES


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: Optional[Path] = None,
    split_name: str = "test"
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates model on dataloader.
    Returns:
      - metrics dictionary
      - y_true array
      - y_pred array
      - y_probs array (N, 7)
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Calculate overall metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)

    per_class_dict = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class_dict[cname] = {
            "name": CLASS_FULL_NAMES[cname],
            "precision": float(per_class_precision[idx]),
            "recall": float(per_class_recall[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(np.sum(y_true == idx))
        }

    metrics = {
        "split": split_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_precision": round(macro_precision, 4),
        "per_class": per_class_dict
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics json
        with open(save_dir / f"metrics_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Generate & save confusion matrix plot
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=CLASS_NAMES,
            save_path=save_dir / f"confusion_matrix_{split_name}.png",
            title=f"Confusion Matrix ({split_name.capitalize()} Set)"
        )

    return metrics, y_true, y_pred, y_probs


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True
):
    """
    Plots and optionally saves a beautiful heatmap confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
        data_to_plot = cm_norm
        fmt = ""
    else:
        data_to_plot = cm
        annot = True
        fmt = "d"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        data_to_plot,
        annot=annot,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Test confusion matrix plotting
    y_true_dummy = np.random.randint(0, 7, size=100)
    y_pred_dummy = np.random.randint(0, 7, size=100)
    plot_confusion_matrix(y_true_dummy, y_pred_dummy, CLASS_NAMES, title="Test Plot")
    print("evaluate.py smoke test passed!")
