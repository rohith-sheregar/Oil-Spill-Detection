"""
Module 2 — Classifier Training Script
Generates training data by running Module 1 on the SOS dataset images,
extracts features from the resulting masks, and trains the Random Forest
look-alike rejection classifier.

Since we don't have manually labelled look-alike images, synthetic negative
samples are generated from the background (inverted) regions of real oil images.
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image

import config
from src.pipeline import load_module1, run_module1
from src.features import extract_features, features_to_array
from src.classifier import LookAlikeClassifier


def generate_training_data(
    img_dir:     str,
    mask_dir:    str,
    label:       int,
    model,
    device:      str,
    max_samples: int = 500,
    invert_mask: bool = False,
) -> tuple:
    """
    Generate feature vectors by running Module 1 on each image and extracting
    patch features.  All patches get the same label.

    Args:
        img_dir:     Directory containing SAR image files.
        mask_dir:    Directory containing ground-truth binary mask files.
        label:       Class label for all extracted patches (1=oil, 0=look_alike).
        model:       Loaded Module 1 model (already on device, eval mode).
        device:      Device string.
        max_samples: Stop early once this many patches have been collected.
        invert_mask: If True, invert the predicted mask before feature extraction
                     (used to sample background / look-alike regions).

    Returns:
        Tuple (X: np.ndarray shape (N,13), y: np.ndarray shape (N,))
    """
    valid_exts = (".png", ".jpg", ".tif", ".tiff")
    image_files = sorted([
        f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)
    ])

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []
    collected                = 0

    desc = f"{'OIL' if label == 1 else 'LOOKALIKE'} samples"
    for fname in tqdm(image_files, desc=desc, leave=True):
        if collected >= max_samples:
            break

        img_path = os.path.join(img_dir, fname)

        # Run Module 1 to get the predicted binary mask
        try:
            original_image, binary_mask = run_module1(model, img_path, device)
        except Exception as e:
            print(f"  ⚠️ Skipping {fname}: {e}")
            continue

        if invert_mask:
            # Use the background region as look-alike negatives
            binary_mask = 1.0 - binary_mask

        # Extract patch features
        feats = extract_features(binary_mask, original_image)
        if not feats:
            continue

        arr = features_to_array(feats)   # (K, 13)
        X_list.append(arr)
        y_list.extend([label] * len(feats))
        collected += len(feats)

    if not X_list:
        return np.empty((0, 13), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.vstack(X_list)[:max_samples]
    y = np.array(y_list[:max_samples], dtype=np.int64)
    return X, y


def train_classifier(
    oil_img_dir:       str,
    oil_mask_dir:      str,
    lookalike_img_dir: str = None,
    lookalike_mask_dir:str = None,
) -> LookAlikeClassifier:
    """
    Train the Module 2 look-alike rejection classifier.

    If lookalike dirs are not provided, synthetic negatives are produced by
    running Module 1 on the same oil images but extracting background patches
    (inverted mask).

    Args:
        oil_img_dir:        Directory of real oil spill SAR images.
        oil_mask_dir:       Corresponding mask directory.
        lookalike_img_dir:  (Optional) Directory of look-alike SAR images.
        lookalike_mask_dir: (Optional) Corresponding mask directory.

    Returns:
        Trained LookAlikeClassifier instance (also saved to disk).
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    # ── Load Module 1 ─────────────────────────────────────────────────────────
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    print(f"Loading Module 1 from: {ckpt_path}")
    model = load_module1(ckpt_path, device)

    # ── Generate positive samples (oil spill) ─────────────────────────────────
    print("\n[1/2] Generating POSITIVE samples (oil spill)…")
    X_pos, y_pos = generate_training_data(
        oil_img_dir, oil_mask_dir, label=1, model=model, device=device,
        max_samples=500
    )
    print(f"  → Collected {len(X_pos)} positive patch features")

    # ── Generate negative samples (look-alike / background) ───────────────────
    print("\n[2/2] Generating NEGATIVE samples (look-alike)…")
    if lookalike_img_dir and lookalike_mask_dir:
        X_neg, y_neg = generate_training_data(
            lookalike_img_dir, lookalike_mask_dir, label=0,
            model=model, device=device, max_samples=500
        )
        print(f"  → Collected {len(X_neg)} look-alike patch features (real look-alike data)")
    else:
        print("  (No look-alike data provided — using synthetic negatives from background regions)")
        X_neg, y_neg = generate_training_data(
            oil_img_dir, oil_mask_dir, label=0,
            model=model, device=device, max_samples=500, invert_mask=True
        )
        print(f"  → Collected {len(X_neg)} synthetic look-alike patch features")

    # ── Combine ───────────────────────────────────────────────────────────────
    if len(X_pos) == 0 or len(X_neg) == 0:
        raise RuntimeError(
            "Not enough data to train — check your data directories and Module 1 checkpoint."
        )

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    print(f"\nClass balance:")
    print(f"  Oil spill  (1): {int(y.sum())} samples")
    print(f"  Look-alike (0): {int((y == 0).sum())} samples")
    print(f"  Total          : {len(y)} samples\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = LookAlikeClassifier(n_estimators=200, random_state=42)
    results = clf.train(X, y)

    print(f"Training accuracy: {results['accuracy']:.4f}\n")
    print(results["report"])
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print()
    print(clf.get_feature_importance_report())

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(config.CHECKPOINT_DIR, "module2_classifier.pkl")
    clf.save(save_path)

    return clf


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 2 — Look-alike Classifier Training")
    print("=" * 60)
    print()

    trained_clf = train_classifier(
        oil_img_dir=config.TRAIN_IMG_DIR,
        oil_mask_dir=config.TRAIN_MASK_DIR,
        # Set below if you have dedicated look-alike imagery:
        # lookalike_img_dir=...,
        # lookalike_mask_dir=...,
    )

    print("\nModule 2 classifier training complete")
    print()
    print(trained_clf.get_feature_importance_report())
