"""
train.py
Main training loop with validation tracking, checkpointing, and history logging.
Optimized for PyTorch with Automatic Mixed Precision (AMP) and CUDA acceleration.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_data_loaders, CLASS_NAMES
from evaluate import evaluate_model
from losses import get_loss_fn
from models.resnet import get_model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch, returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(train_loader)
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"      Batch [{batch_idx+1:03d}/{total_batches:03d}] Loss: {loss.item():.4f} - Running Acc: {correct/total*100:.1f}%", flush=True)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def train_experiment(
    model_name: str = "resnet50",
    loss_type: str = "ce",
    sampling_mode: str = "random",
    aug_mode: str = "basic",
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    save_dir: Union[str, Path] = "results/baseline",
    data_dir: Union[str, Path] = "data",
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Executes a complete training and evaluation pipeline for a single experiment configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    print(f"\n{'='*70}")
    print(f"[*] Starting Experiment: {save_dir.name}")
    print(f"    - Model: {model_name}")
    print(f"    - Loss: {loss_type}")
    print(f"    - Sampling: {sampling_mode}")
    print(f"    - Augmentation: {aug_mode}")
    print(f"    - Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"    - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'='*70}")

    # 1. Load Data
    train_loader, val_loader, test_loader, train_class_counts, _ = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        aug_mode=aug_mode,
        sampling_mode=sampling_mode,
        seed=seed
    )

    # 2. Build Model
    model = get_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)

    # 3. Setup Loss & Optimizer
    criterion = get_loss_fn(loss_type, class_counts=train_class_counts, device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_macro_f1 = -1.0
    history = []

    # 4. Training Loop
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validation evaluation
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device, split_name="val")
        scheduler.step()
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        val_macro_f1 = val_metrics['macro_f1']
        val_acc = val_metrics['accuracy']
        val_bal_acc = val_metrics['balanced_accuracy']

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% - Val BalAcc: {val_bal_acc*100:.2f}% - Val MacroF1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": val_acc,
            "val_balanced_acc": val_bal_acc,
            "val_macro_f1": val_macro_f1,
            "val_macro_recall": val_metrics['macro_recall'],
            "lr": current_lr,
            "time_sec": round(epoch_time, 2)
        })

        # Save checkpoint based on Validation Macro F1 (Key metric)
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_metrics": val_metrics,
                "config": {
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "sampling_mode": sampling_mode,
                    "aug_mode": aug_mode
                }
            }, save_dir / "best_model.pth")
            print(f"  [+] Saved new best model checkpoint (Val Macro F1: {best_val_macro_f1:.4f})", flush=True)

    # 5. Save History
    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "training_history.csv", index=False)

    # 6. Final Evaluation on Test Set using Best Checkpoint
    print(f"\n[*] Evaluating best model checkpoint on Test Set...", flush=True)
    best_checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics, _, _, _ = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        save_dir=save_dir,
        split_name="test"
    )

    print(f"[+] Test Set Evaluation Complete for {save_dir.name}:")
    print(f"    - Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"    - Balanced Accuracy: {test_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    - Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(f"    - Weighted F1:       {test_metrics['weighted_f1']:.4f}")

    return {
        "config": {
            "model_name": model_name,
            "loss_type": loss_type,
            "sampling_mode": sampling_mode,
            "aug_mode": aug_mode,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr
        },
        "best_val_macro_f1": best_val_macro_f1,
        "test_metrics": test_metrics
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Skin-Lesion Classification Model")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "weighted_ce", "focal", "cb_loss"])
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "weighted", "balanced"])
    parser.add_argument("--aug", type=str, default="basic", choices=["basic", "advanced"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--save_dir", type=str, default="results/baseline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_experiment(
        model_name=args.model,
        loss_type=args.loss,
        sampling_mode=args.sampling,
        aug_mode=args.aug,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        seed=args.seed
    )
