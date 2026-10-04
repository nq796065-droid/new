# DL2026-Group29: Skin-Lesion Classification under Severe Class Imbalance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system investigating and mitigating the severe class imbalance problem in skin lesion classification across 7 dermatoscopic categories (HAM10000 benchmark).

---

## 📌 Project Overview
- **Project ID**: 29
- **Course**: Deep Learning Final Project
- **Primary Objective**: Build a ResNet-50 dermatoscopic classifier and systematically benchmark strategies to mitigate extreme class imbalance (NV majority at 66.95% vs DF minority at 1.15%), focusing on **Macro F1** and **Balanced Accuracy**.
- **Key Methods Investigated**:
  1. **Sampling**: Random vs. Weighted Random Sampling (Inverse Class Frequency)
  2. **Loss Functions**: Cross-Entropy vs. Weighted CE vs. Focal Loss ($\gamma = 2.0$)
  3. **Data Augmentation**: Basic Geometric vs. Advanced Photometric & Spatial (Color Jitter, Affine)
  4. **Combined Strategy**: Synergistic integration of Weighted Sampling, Focal Loss, and Advanced Augmentation.

---

## 📂 Project Structure
```text
skin-lesion-classification/
├── data/                       # HAM10000 images, metadata, and splits.csv
│   ├── HAM10000_metadata.csv
│   ├── splits.csv              # Leakage-free Stratified Group split (Seed 42)
│   └── images/                 # 10,015 JPG dermatoscopic images
├── models/
│   ├── __init__.py
│   └── resnet.py               # ResNet-50 and EfficientNet-B0 architectures
├── results/                    # Checkpoints, metrics, and figures
│   ├── baseline/               # Baseline experiment artifacts
│   ├── weighted_sampling/      # Sampling ablation artifacts
│   ├── focal_loss/             # Focal loss ablation artifacts
│   ├── augmentation/           # Advanced augmentation artifacts
│   ├── combined/               # Combined best strategy artifacts
│   ├── comparison.csv          # Aggregate metrics across all setups
│   ├── comparison_chart.png    # Multi-metric visual comparison
│   └── per_class_recall_comparison.png # Minority class gain visualization
├── data.py                     # DataLoaders, transforms, patient-aware splitting
├── train.py                    # Training engine with AMP, LR scheduling & checkpointing
├── losses.py                   # CE, Weighted CE, Focal Loss, Class-Balanced Loss
├── evaluate.py                 # Multi-class evaluation metrics & confusion matrix
├── experiments.py              # Automated 5-experiment benchmark runner
├── demo.py                     # Interactive Gradio web application for clinical inference
├── DATA.md                     # Official dataset metadata & reproduction docs
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone repository & create virtual environment
```bash
git clone https://github.com/YourGroup/DL2026-Group29-Project29.git
cd DL2026-Group29-Project29

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation
The project utilizes the **HAM10000** dataset (10,015 images). To download and prepare the zero-leakage splits:
```bash
python download_dataset.py
```
*Note: Metadata contains `lesion_id`. Splits are strictly created via `StratifiedGroupKFold` so that multiple images of the same lesion never cross between train, validation, or test sets.*

---

## 🚀 Running Experiments & Reproducing Results

### 1. Train Individual Model (e.g. Baseline)
```bash
python train.py --model resnet50 --loss ce --sampling random --aug basic --epochs 15 --save_dir results/baseline
```

### 2. Train with Focal Loss
```bash
python train.py --model resnet50 --loss focal --sampling random --aug basic --epochs 15 --save_dir results/focal_loss
```

### 3. Train with Weighted Random Sampling
```bash
python train.py --model resnet50 --loss ce --sampling weighted --aug basic --epochs 15 --save_dir results/weighted_sampling
```

### 4. Run the Full 5-Experiment Benchmark Suite
To automatically execute all 5 ablation experiments, log training histories, evaluate test sets, and generate `comparison.csv` and charts:
```bash
python experiments.py --epochs 15 --batch_size 32 --lr 0.0001
```

---

## 📈 Experimental Results Summary

| Experiment Configuration | Loss Function | Sampling Strategy | Data Augmentation | Accuracy | Balanced Accuracy | Macro F1 (Key) | Macro Recall |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Cross-Entropy | Random | Basic Flip | 80.6% | 58.2% | 0.6012 | 58.2% |
| **Sampling Ablation** | Cross-Entropy | Weighted Sampler | Basic Flip | 76.4% | 71.5% | 0.6720 | 71.5% |
| **Loss Ablation** | Focal Loss ($\gamma=2$) | Random | Basic Flip | 81.2% | 63.8% | 0.6485 | 63.8% |
| **Augmentation Ablation** | Cross-Entropy | Random | Color Jitter + Affine | 81.8% | 61.4% | 0.6350 | 61.4% |
| **Combined Strategy** | **Focal Loss** | **Weighted Sampler** | **Advanced Aug** | **79.5%** | **75.8%** | **0.7240** | **75.8%** |

*Key Finding: While naive Baseline achieves high overall accuracy by prioritizing the dominant class (`nv`), the Combined Strategy dramatically boosts minority class detection—raising Balanced Accuracy by +17.6% and Macro F1 by +0.1228.*

---

## 🩺 Interactive Clinical Demo
Launch the Gradio web application for single-image diagnosis, class probability distribution, and clinical risk stratification:
```bash
python demo.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## ⚠️ Academic Disclaimer
This system is an academic research demonstration for Deep Learning coursework. It is **not** certified as a clinical medical device. All outputs must be verified by licensed medical professionals.
