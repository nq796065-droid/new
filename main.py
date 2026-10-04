"""
demo.py
Interactive Gradio Web Application for Skin-Lesion Classification and Clinical Risk Assessment.
Loads trained ResNet-50 weights, computes class probabilities, risk stratification, and visualizes predictions.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from data import CLASS_NAMES, CLASS_FULL_NAMES, IMAGENET_MEAN, IMAGENET_STD
from models.resnet import get_model


# Medical risk categories
HIGH_RISK_CLASSES = ["mel", "bcc"]
MODERATE_RISK_CLASSES = ["akiec"]
BENIGN_CLASSES = ["nv", "bkl", "df", "vasc"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = Path("results/weighted_sampling/best_model.pth")
FALLBACK_MODEL_PATH = Path("results/focal_loss/best_model.pth")
BASELINE_MODEL_PATH = Path("results/baseline/best_model.pth")

# Preprocessing pipeline
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Global model instance
model = None


def load_model(checkpoint_path: Path):
    global model
    model = get_model(model_name="resnet50", num_classes=len(CLASS_NAMES), pretrained=False)
    if checkpoint_path.exists():
        print(f"[*] Loading model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}. Using initial weights.")
    model.to(DEVICE)
    model.eval()


def predict_skin_lesion(image: Image.Image) -> Tuple[Dict[str, float], str, str]:
    """
    Takes an input PIL Image and returns:
      1. Dictionary of probabilities per class
      2. Clinical Risk Level badge and description
      3. Top diagnostic recommendation
    """
    global model
    if model is None:
        ckpt = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
        load_model(ckpt)

    if image is None:
        return {}, "No image uploaded", ""

    # Preprocess
    img_rgb = image.convert("RGB")
    input_tensor = eval_transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().squeeze(0).numpy()

    # Create probability dictionary with readable names
    prob_dict = {
        f"{CLASS_FULL_NAMES[cname]} ({cname.upper()})": float(probs[idx])
        for idx, cname in enumerate(CLASS_NAMES)
    }

    # Top prediction
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_confidence = probs[top_idx] * 100.0

    # Risk analysis
    mel_risk = probs[CLASS_NAMES.index("mel")] * 100.0
    bcc_risk = probs[CLASS_NAMES.index("bcc")] * 100.0

    if top_class in HIGH_RISK_CLASSES or mel_risk > 20.0 or bcc_risk > 25.0:
        risk_badge = "🔴 HIGH RISK (Malignancy Suspicion)"
        risk_desc = (
            f"**High Priority Referral Recommended.**\n\n"
            f"- Primary Prediction: **{CLASS_FULL_NAMES[top_class]} ({top_class.upper()})** with **{top_confidence:.1f}%** confidence.\n"
            f"- Melanoma (MEL) probability: **{mel_risk:.1f}%**\n"
            f"- Basal Cell Carcinoma (BCC) probability: **{bcc_risk:.1f}%**\n\n"
            f"⚠️ *Urgent dermatoscopic examination or biopsy is strongly advised.*"
        )
    elif top_class in MODERATE_RISK_CLASSES:
        risk_badge = "🟡 MODERATE RISK (Pre-cancerous / In-situ)"
        risk_desc = (
            f"**Dermatological Follow-up Advised.**\n\n"
            f"- Primary Prediction: **{CLASS_FULL_NAMES[top_class]} ({top_class.upper()})** with **{top_confidence:.1f}%** confidence.\n\n"
            f"ℹ️ *Actinic keratoses may possess malignant potential and should be medically assessed.*"
        )
    else:
        risk_badge = "🟢 LOW RISK (Benign Indication)"
        risk_desc = (
            f"**Probable Benign Lesion.**\n\n"
            f"- Primary Prediction: **{CLASS_FULL_NAMES[top_class]} ({top_class.upper()})** with **{top_confidence:.1f}%** confidence.\n\n"
            f"✅ *Lesion exhibits characteristics typical of benign tissue ({CLASS_FULL_NAMES[top_class]}). Monitor for ABCDE changes.*"
        )

    disclaimer = (
        "### ⚠️ Academic Research Disclaimer\n"
        "This artificial intelligence tool is developed solely for academic research and educational purposes. "
        "It is **not** a certified medical diagnostic device and **must never** be used as a substitute for professional clinical diagnosis or medical decision-making."
    )

    return prob_dict, f"{risk_badge}\n\n{risk_desc}", disclaimer


def build_app():
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .risk-box { padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; background-color: #f9f9f9; }
    """

    with gr.Blocks(title="Skin Lesion AI Classifier | HAM10000", css=custom_css) as demo:
        gr.Markdown(
            "# 🔬 Skin-Lesion Classification System\n"
            "### Deep Learning under Severe Class Imbalance (HAM10000 Benchmark)\n"
            "*Powered by ResNet-50 with Focal Loss, Weighted Sampling & Data Augmentation*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Dermatoscopic Skin Image")
                submit_btn = gr.Button("Analyze Lesion", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=7, label="Class Probabilities")
                risk_output = gr.Markdown(label="Clinical Stratification")

        disclaimer_output = gr.Markdown()

        submit_btn.click(
            fn=predict_skin_lesion,
            inputs=[image_input],
            outputs=[label_output, risk_output, disclaimer_output]
        )

    return demo


if __name__ == "__main__":
    ckpt = DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    load_model(ckpt)
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
