"""
Module 2 — End-to-End SAR Oil Spill Pipeline
Chains Module 1 (DeepLabV3+ segmentation) → Module 2 (look-alike rejection)
on a single SAR image and produces a colour-coded visualisation.
"""

import os
import sys
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from src.model import OilSpillDeepLab
from src.utils import load_checkpoint
from src.dataset import get_transforms
from src.features import extract_features, describe_patch
from src.classifier import LookAlikeClassifier


# ── Module 1 helpers ─────────────────────────────────────────────────────────

def load_module1(checkpoint_path: str, device: str) -> OilSpillDeepLab:
    """
    Instantiate the DeepLabV3+ model and load pre-trained weights.

    Args:
        checkpoint_path: Path to best_model.pth.
        device:          Target device string ('cuda' or 'cpu').

    Returns:
        Loaded model set to eval mode.
    """
    model = OilSpillDeepLab(num_classes=1).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


def run_module1(
    model: OilSpillDeepLab,
    image_path: str,
    device: str,
) -> tuple:
    """
    Run Module 1 inference on a single SAR image.

    Args:
        model:      Loaded OilSpillDeepLab model.
        image_path: Path to the input SAR image.
        device:     Device string.

    Returns:
        Tuple of:
            original_image_np — uint8 numpy array of shape (H, W) (grayscale)
            binary_mask_np    — float32 numpy array of shape (H, W) with values 0/1
    """
    # Load image and resize to model input size
    pil_img = Image.open(image_path).convert("RGB").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    )
    img_np = np.array(pil_img)  # (H, W, 3), uint8

    # Use the same albumentations pipeline the model was trained with
    dummy_mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
    aug = get_transforms(train=False)(image=img_np, mask=dummy_mask)
    tensor = aug["image"].unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # (H, W)

    binary_mask = (prob > 0.5).astype(np.float32)

    # Return original grayscale image (first channel) for visualisation
    original_gray = img_np[:, :, 0]  # (H, W), uint8

    return original_gray, binary_mask


# ── Module 2 helpers ─────────────────────────────────────────────────────────

def run_module2(
    mask: np.ndarray,
    image: np.ndarray,
    classifier: LookAlikeClassifier,
    acquisition_hour: int = 2,
) -> tuple:
    """
    Run look-alike rejection (Module 2) on the output of Module 1.

    Args:
        mask:             Binary float32 array of shape (H, W).
        image:            Grayscale image array of shape (H, W).
        classifier:       LookAlikeClassifier instance (trained or untrained).
        acquisition_hour: UTC hour of SAR acquisition.

    Returns:
        Tuple of (features: list[PatchFeatures], predictions: list[dict]).
        If classifier is untrained, predictions is an empty list and a warning
        is printed.
    """
    features = extract_features(mask, image, acquisition_hour=acquisition_hour)

    if not classifier.is_trained:
        warnings.warn(
            "⚠️  Module 2 classifier is not trained. "
            "Skipping look-alike rejection. Run train_module2.py first."
        )
        return features, []

    predictions = classifier.predict(features)
    return features, predictions


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_pipeline(
    image: np.ndarray,
    mask: np.ndarray,
    features: list,
    predictions: list,
    save_path: str = None,
) -> None:
    """
    Produce a 3-panel figure comparing SAR input, Module 1 mask, and Module 2 result.

    Panel 0 — SAR Input (grayscale).
    Panel 1 — Module 1 detection: red overlay on all detected pixels.
    Panel 2 — Module 2 result:
               • green  overlay → confirmed oil_spill patches
               • orange overlay → rejected look_alike patches.

    Args:
        image:      Grayscale image array (H, W) uint8.
        mask:       Binary mask array (H, W) float32.
        features:   List of PatchFeatures from extract_features().
        predictions:List of prediction dicts from LookAlikeClassifier.predict().
        save_path:  If given, save the figure here at 150 dpi.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Oil Spill Pipeline — Module 1 Detection → Module 2 Classification",
        fontsize=13, fontweight="bold",
    )

    gray_float = image.astype(np.float32) / 255.0

    # ── Panel 0: SAR input ────────────────────────────────────────────────────
    axes[0].imshow(gray_float, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("SAR Input", fontweight="bold")
    axes[0].axis("off")

    # ── Panel 1: Module 1 detection (red overlay) ─────────────────────────────
    axes[1].imshow(gray_float, cmap="gray", vmin=0, vmax=1)
    overlay_m1 = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay_m1[mask == 1] = [1.0, 0.0, 0.0, 0.55]
    axes[1].imshow(overlay_m1)
    axes[1].set_title("Module 1 — Detected Patches", fontweight="bold")
    axes[1].axis("off")

    # ── Panel 2: Module 2 result ──────────────────────────────────────────────
    axes[2].imshow(gray_float, cmap="gray", vmin=0, vmax=1)

    # Build per-pixel coloured overlay by rasterising each connected component
    overlay_m2 = np.zeros((*mask.shape, 4), dtype=np.float32)
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    # Build valid component list matching what extract_features produced
    valid_components = []
    for lid in range(1, num_labels):
        if stats[lid, cv2.CC_STAT_AREA] >= 50:
            valid_components.append(lid)

    for idx, comp_id in enumerate(valid_components):
        comp_mask = labels_map == comp_id
        if idx < len(predictions):
            pred = predictions[idx]
            if pred["is_oil"]:
                colour = [0.0, 0.8, 0.2, 0.55]  # green
            else:
                colour = [1.0, 0.55, 0.0, 0.55]  # orange
            # Add confidence annotation at component centroid
            ys, xs = np.where(comp_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            label_txt = f"{pred['label'][:3].upper()}\n{pred['confidence']:.2f}"
            axes[2].text(
                cx, cy, label_txt,
                color="white", fontsize=6, ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0),
            )
        else:
            colour = [1.0, 0.0, 0.0, 0.55]  # red if no prediction

        overlay_m2[comp_mask] = colour

    axes[2].imshow(overlay_m2)
    axes[2].set_title("Module 2 — Look-alike Rejection", fontweight="bold")
    axes[2].axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=(1, 0, 0, 0.7),     label="M1 Detection"),
        mpatches.Patch(color=(0, 0.8, 0.2, 0.7), label="Confirmed Oil"),
        mpatches.Patch(color=(1, 0.55, 0, 0.7),  label="Look-alike Rejected"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Visualization saved → {save_path}")

    plt.show()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    classifier_path: str = None,
    acquisition_hour: int = 2,
    save_viz: bool = True,
) -> dict:
    """
    Execute the full Module 1 → Module 2 pipeline on a single SAR image.

    Args:
        image_path:       Path to the input SAR image file.
        classifier_path:  Optional path to a saved Module 2 classifier .pkl.
        acquisition_hour: UTC hour of image acquisition (used for night feature).
        save_viz:         If True, save the visualisation figure.

    Returns:
        Dict with keys:
            mask, features, predictions, n_detected, n_confirmed_oil, n_rejected
    """
    device = config.DEVICE

    # ── Step 1: Load Module 1 ─────────────────────────────────────────────────
    print("🛰️  Loading Module 1 (DeepLabV3+ segmentation model)…")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model = load_module1(ckpt_path, device)

    # ── Step 2: Run Module 1 segmentation ────────────────────────────────────
    print(f"🔍  Running Module 1 inference on: {image_path}")
    original_image, binary_mask = run_module1(model, image_path, device)
    n_detected_pixels = int(binary_mask.sum())
    print(f"    → Detected {n_detected_pixels} oil pixels in the mask")

    # ── Step 3: Load Module 2 classifier ─────────────────────────────────────
    # Default to the standard saved location if no path is given
    if classifier_path is None:
        classifier_path = os.path.join(config.CHECKPOINT_DIR, "module2_classifier.pkl")

    classifier = LookAlikeClassifier()
    if os.path.exists(classifier_path):
        print(f"🌊  Loading Module 2 classifier from: {classifier_path}")
        classifier.load(classifier_path)
    else:
        print(f"⚠️   Classifier not found at '{classifier_path}' — running without M2")
        print("     (Run 'python -m src.train_module2' to train it first)")

    # ── Step 4: Run Module 2 look-alike rejection ─────────────────────────────
    print("🌊  Running Module 2 look-alike rejection…")
    features, predictions = run_module2(
        binary_mask, original_image, classifier, acquisition_hour
    )
    n_detected = len(features)

    # ── Step 5: Per-patch summary ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Detected {n_detected} connected patch(es):")
    n_confirmed_oil = 0
    n_rejected      = 0

    for i, feat in enumerate(features):
        desc = describe_patch(feat)
        if i < len(predictions):
            pred = predictions[i]
            status = "✅ OIL" if pred["is_oil"] else "❌ LOOK-ALIKE"
            conf   = f"(conf: {pred['confidence']:.3f})"
            if pred["is_oil"]:
                n_confirmed_oil += 1
            else:
                n_rejected += 1
        else:
            status = "❓ UNCLASSIFIED"
            conf   = ""
            n_confirmed_oil += 1  # treat unclassified as potential oil for safety
        print(f"  Patch {i+1}: {status} {conf} | {desc}")

    print(f"{'─'*60}")
    print(f"  Summary: {n_confirmed_oil} confirmed oil | {n_rejected} rejected")
    print(f"{'─'*60}\n")

    # ── Step 6: Visualise ─────────────────────────────────────────────────────
    if save_viz:
        save_path = os.path.join(config.PRED_DIR, "pipeline_result.png")
    else:
        save_path = None

    visualize_pipeline(original_image, binary_mask, features, predictions, save_path)

    return {
        "image":            original_image,
        "mask":             binary_mask,
        "features":         features,
        "predictions":      predictions,
        "n_detected":       n_detected,
        "n_confirmed_oil":  n_confirmed_oil,
        "n_rejected":       n_rejected,
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.pipeline <image_path> [classifier_path]\n"
            "Example:\n"
            "  python -m src.pipeline data/sos/test/sentinel/image/img001.png\n"
            "  python -m src.pipeline data/sos/test/sentinel/image/img001.png "
            "outputs/checkpoints/module2_classifier.pkl"
        )
        sys.exit(1)

    _image_path      = sys.argv[1]
    _classifier_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_pipeline(_image_path, _classifier_path)
    print("\nFinal result summary:")
    for k, v in result.items():
        if k not in ("mask", "features", "predictions"):
            print(f"  {k}: {v}")
