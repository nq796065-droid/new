"""
experiments.py
Runs the structured suite of 5 benchmark experiments comparing imbalance mitigation techniques:
  1. Baseline: ResNet-50 + Random Sampling + Cross-Entropy + Basic Augmentation
  2. Sampling: ResNet-50 + Weighted Random Sampling + Cross-Entropy + Basic Augmentation
  3. Loss Function: ResNet-50 + Random Sampling + Focal Loss + Basic Augmentation
  4. Augmentation: ResNet-50 + Random Sampling + Cross-Entropy + Advanced Augmentation
  5. Combined: ResNet-50 + Weighted Sampling + Focal Loss + Advanced Augmentation

Aggregates results into comparison.csv and generates publication-grade comparison charts.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import CLASS_NAMES, CLASS_FULL_NAMES
from train import train_experiment


EXPERIMENT_CONFIGS = [
    {
        "name": "baseline",
        "description": "Baseline: ResNet-50 + CE + Random Sampling + Basic Aug",
        "model_name": "resnet50",
        "loss_type": "ce",
        "sampling_mode": "random",
        "aug_mode": "basic"
    },
    {
        "name": "weighted_sampling",
        "description": "Sampling: Weighted Random Sampling",
        "model_name": "resnet50",
        "loss_type": "ce",
        "sampling_mode": "weighted",
        "aug_mode": "basic"
    },
    {
        "name": "focal_loss",
        "description": "Loss Function: Focal Loss (gamma=2.0)",
        "model_name": "resnet50",
        "loss_type": "focal",
        "sampling_mode": "random",
        "aug_mode": "basic"
    },
    {
        "name": "augmentation",
        "description": "Augmentation: Color Jitter, Rotation, Affine",
        "model_name": "resnet50",
        "loss_type": "ce",
        "sampling_mode": "random",
        "aug_mode": "advanced"
    },
    {
        "name": "combined",
        "description": "Combined Best: Weighted Sampling + Focal Loss + Advanced Aug",
        "model_name": "resnet50",
        "loss_type": "focal",
        "sampling_mode": "weighted",
        "aug_mode": "advanced"
    }
]


def plot_comparison_charts(summary_df: pd.DataFrame, results_dir: Path):
    """
    Generates multi-metric comparison bar charts and per-class recall comparisons.
    """
    sns.set_theme(style="whitegrid")
    results_dir = Path(results_dir)

    # 1. Macro-level metrics comparison
    metrics_to_plot = ["accuracy", "balanced_accuracy", "macro_f1", "macro_recall"]
    metric_labels = ["Accuracy", "Balanced Accuracy", "Macro F1 (Key)", "Macro Recall"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    palette = sns.color_palette("deep", len(summary_df))

    for idx, (col, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        bars = ax.bar(summary_df["Experiment"], summary_df[col] * 100, color=palette)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticklabels(summary_df["Experiment"], rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    axes[0].set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    plt.suptitle("Comparative Evaluation Across Imbalance Mitigation Strategies (HAM10000 Test Set)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    chart_path = results_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart to {chart_path}")

    # 2. Per-class recall heatmap across experiments
    # Read all metrics_test.json
    per_class_matrix = []
    exp_names = []

    for cfg in EXPERIMENT_CONFIGS:
        exp_dir = results_dir / cfg["name"]
        metrics_file = exp_dir / "metrics_test.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
            recalls = [data["per_class"][c]["recall"] * 100 for c in CLASS_NAMES]
            per_class_matrix.append(recalls)
            exp_names.append(cfg["name"])

    if per_class_matrix:
        matrix_df = pd.DataFrame(per_class_matrix, index=exp_names, columns=[c.upper() for c in CLASS_NAMES])
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Recall (%)'})
        plt.title("Per-Class Recall Comparison Across Experiments (Showing Minority Class Gains)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Lesion Class", fontsize=11, fontweight="bold")
        plt.ylabel("Experiment", fontsize=11, fontweight="bold")
        plt.tight_layout()
        per_class_chart = results_dir / "per_class_recall_comparison.png"
        plt.savefig(per_class_chart, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved per-class recall chart to {per_class_chart}")


def run_all_experiments(
    results_dir: Path = Path("results"),
    data_dir: Path = Path("data"),
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42
):
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for cfg in EXPERIMENT_CONFIGS:
        exp_name = cfg["name"]
        exp_dir = results_dir / exp_name

        print(f"\n=======================================================")
        print(f"  RUNNING BENCHMARK: {exp_name.upper()}")
        print(f"=======================================================")

        res = train_experiment(
            model_name=cfg["model_name"],
            loss_type=cfg["loss_type"],
            sampling_mode=cfg["sampling_mode"],
            aug_mode=cfg["aug_mode"],
            num_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=exp_dir,
            data_dir=data_dir,
            seed=seed
        )

        test_m = res["test_metrics"]
        summary_records.append({
            "Experiment": exp_name,
            "Description": cfg["description"],
            "Loss": cfg["loss_type"],
            "Sampling": cfg["sampling_mode"],
            "Augmentation": cfg["aug_mode"],
            "accuracy": test_m["accuracy"],
            "balanced_accuracy": test_m["balanced_accuracy"],
            "macro_f1": test_m["macro_f1"],
            "weighted_f1": test_m["weighted_f1"],
            "macro_recall": test_m["macro_recall"],
            "macro_precision": test_m["macro_precision"],
            "val_macro_f1_best": round(res["best_val_macro_f1"], 4)
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = results_dir / "comparison.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[+] Successfully saved comparison table to {summary_csv}")

    # Plot comparison charts
    plot_comparison_charts(summary_df, results_dir)

    # Automatically generate 4-slide defense presentation and comprehensive 10-15 page PDF report
    try:
        from generate_slides import build_slides
        build_slides(results_dir / "Group29_Project29_Slides.pptx")
        print("[+] Generated 4-slide presentation.")
    except Exception as e:
        print(f"[!] Warning generating slides: {e}")

    try:
        from generate_report import generate_pdf_report
        generate_pdf_report(str(results_dir / "Group29_Project29_Report.pdf"), charts_dir=results_dir)
        print("[+] Generated formal academic report PDF.")
    except Exception as e:
        print(f"[!] Warning generating report: {e}")

    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE:")
    print("="*80)
    print(summary_df[["Experiment", "accuracy", "balanced_accuracy", "macro_f1", "macro_recall"]].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_experiments(
        results_dir=Path(args.results_dir),
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
