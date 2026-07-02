"""
Script to train Module 2 (Random Forest look-alike classifier).

Uses the MKLab dataset by default, which has real oil_spill and look_alike labels.
Pass --no-mklab to fall back to synthetic negatives from the SOS dataset.

Usage:
  python train_module2.py                # MKLab real labels (recommended)
  python train_module2.py --no-mklab     # SOS synthetic negatives (legacy)
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import config
from src.pipeline import load_module1, run_module1
from src.features import extract_features, features_to_array
from src.classifier import LookAlikeClassifier



# ── MKLab-based data generation (real labels) ────────────────────────────────

def generate_mklab_training_data(
    img_dir:   str,
    label_dir: str,
    max_oil:   int = 500,
    max_look:  int = 500,
) -> tuple:
    """
    Generate feature vectors from MKLab ground-truth labels.

    Uses the 1D label masks directly:
      - Class 1 (oil_spill)  -> positive samples (label=1)
      - Class 2 (look_alike) -> negative samples (label=0)

    No Module 1 inference is needed -- we use the real segmentation masks.

    Args:
        img_dir:   Directory of MKLab SAR images (.jpg).
        label_dir: Directory of MKLab 1D label masks (.png).
        max_oil:   Maximum oil spill patches to collect.
        max_look:  Maximum look-alike patches to collect.

    Returns:
        Tuple (X: np.ndarray shape (N,13), y: np.ndarray shape (N,))
    """
    valid_exts = (".png", ".jpg", ".tif", ".tiff")
    image_files = sorted([
        f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)
    ])

    X_oil, y_oil = [], []
    X_look, y_look = [], []
    oil_count, look_count = 0, 0

    for fname in tqdm(image_files, desc="MKLab features", leave=True):
        if oil_count >= max_oil and look_count >= max_look:
            break

        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, base + ".png")

        if not os.path.exists(label_path):
            continue

        # Load and resize to match model input
        pil_img = Image.open(img_path).convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
        pil_label = Image.open(label_path).convert("L").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST
        )

        img_np = np.array(pil_img)[:, :, 0]  # grayscale channel
        label_np = np.array(pil_label)

        # ── Oil spill patches (class 1) ──
        if oil_count < max_oil:
            oil_mask = (label_np == 1).astype(np.float32)
            if oil_mask.sum() >= 50:  # min area filter
                feats = extract_features(oil_mask, img_np)
                if feats:
                    arr = features_to_array(feats)
                    X_oil.append(arr)
                    y_oil.extend([1] * len(feats))
                    oil_count += len(feats)

        # ── Look-alike patches (class 2) ──
        if look_count < max_look:
            look_mask = (label_np == 2).astype(np.float32)
            if look_mask.sum() >= 50:  # min area filter
                feats = extract_features(look_mask, img_np)
                if feats:
                    arr = features_to_array(feats)
                    X_look.append(arr)
                    y_look.extend([0] * len(feats))
                    look_count += len(feats)

    # Combine
    all_X = X_oil + X_look
    all_y = y_oil + y_look

    if not all_X:
        return np.empty((0, 13), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# ── SOS-based data generation (synthetic, legacy) ────────────────────────────

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

    X_list = []
    y_list = []
    collected = 0

    desc = f"{'OIL' if label == 1 else 'LOOKALIKE'} samples"
    for fname in tqdm(image_files, desc=desc, leave=True):
        if collected >= max_samples:
            break

        img_path = os.path.join(img_dir, fname)

        try:
            original_image, binary_mask = run_module1(model, img_path, device)
        except Exception as e:
            print(f"  Warning: Skipping {fname}: {e}")
            continue

        if invert_mask:
            binary_mask = 1.0 - binary_mask

        feats = extract_features(binary_mask, original_image)
        if not feats:
            continue

        arr = features_to_array(feats)
        X_list.append(arr)
        y_list.extend([label] * len(feats))
        collected += len(feats)

    if not X_list:
        return np.empty((0, 13), dtype=np.float32), np.empty((0,), dtype=np.int64)

    X = np.vstack(X_list)[:max_samples]
    y = np.array(y_list[:max_samples], dtype=np.int64)
    return X, y


# ── Unified training function ─────────────────────────────────────────────────

def train_classifier(
    oil_img_dir:       str = None,
    oil_mask_dir:      str = None,
    lookalike_img_dir: str = None,
    lookalike_mask_dir:str = None,
    use_mklab:         bool = True,
) -> LookAlikeClassifier:
    """
    Train the Module 2 look-alike rejection classifier.

    Args:
        oil_img_dir:        (SOS mode) Directory of real oil spill SAR images.
        oil_mask_dir:       (SOS mode) Corresponding mask directory.
        lookalike_img_dir:  (SOS mode) Optional directory of look-alike images.
        lookalike_mask_dir: (SOS mode) Optional mask directory.
        use_mklab:          If True, use MKLab dataset with real labels.

    Returns:
        Trained LookAlikeClassifier instance (also saved to disk).
    """
    device = config.DEVICE
    print(f"Using device: {device}")

    if use_mklab:
        # ── MKLab mode: use real ground-truth labels ──────────────────────────
        print("\n[MKLab Mode] Using real oil_spill and look_alike labels...")
        X, y = generate_mklab_training_data(
            img_dir=config.MKLAB_TRAIN_IMG_DIR,
            label_dir=config.MKLAB_TRAIN_LABEL_DIR,
            max_oil=500,
            max_look=500,
        )
    else:
        # ── SOS mode: synthetic negatives (legacy) ────────────────────────────
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        print(f"Loading Module 1 from: {ckpt_path}")
        model = load_module1(ckpt_path, device)

        print("\n[1/2] Generating POSITIVE samples (oil spill)...")
        X_pos, y_pos = generate_training_data(
            oil_img_dir, oil_mask_dir, label=1, model=model, device=device,
            max_samples=500
        )
        print(f"  -> Collected {len(X_pos)} positive patch features")

        print("\n[2/2] Generating NEGATIVE samples (look-alike)...")
        if lookalike_img_dir and lookalike_mask_dir:
            X_neg, y_neg = generate_training_data(
                lookalike_img_dir, lookalike_mask_dir, label=0,
                model=model, device=device, max_samples=500
            )
            print(f"  -> Collected {len(X_neg)} look-alike patch features (real data)")
        else:
            print("  (No look-alike data -- using synthetic negatives from background)")
            X_neg, y_neg = generate_training_data(
                oil_img_dir, oil_mask_dir, label=0,
                model=model, device=device, max_samples=500, invert_mask=True
            )
            print(f"  -> Collected {len(X_neg)} synthetic look-alike patch features")

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

    # ── Validate ──────────────────────────────────────────────────────────────
    if len(X) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError(
            "Not enough data to train -- check your data directories."
        )

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Module 2 look-alike classifier")
    parser.add_argument(
        "--no-mklab", action="store_true",
        help="Disable MKLab data and use SOS synthetic negatives instead"
    )
    args = parser.parse_args()

    use_mklab = not args.no_mklab

    print("=" * 60)
    print("  MODULE 2 -- Look-alike Classifier Training")
    print(f"  Data source: {'MKLab (real labels)' if use_mklab else 'SOS (synthetic)'}")
    print("=" * 60)
    print()

    trained_clf = train_classifier(
        oil_img_dir=config.TRAIN_IMG_DIR,
        oil_mask_dir=config.TRAIN_MASK_DIR,
        use_mklab=use_mklab,
    )

    print("\nModule 2 classifier training complete")
    print()
    print(trained_clf.get_feature_importance_report())
