"""
evaluate.py — Module 1 + Module 2 Combined Evaluation Script

Tests a trained Module 1 checkpoint (segmentation) combined with the
trained Module 2 classifier (look-alike rejection) on the test set.

Supports both model versions:
  --backbone mobilenet  → OilSpillModelV1 (Run 1-4, 256x256)
  --backbone efficientnet → OilSpillModel (Run 5+, 512x512)

Usage:
  # Test old MobileNetV3 checkpoint (default):
  python evaluate.py --backbone mobilenet --checkpoint outputs/checkpoints/best_model.pth

  # Test new EfficientNet-B4 checkpoint:
  python evaluate.py --backbone efficientnet --checkpoint outputs/checkpoints/best_model_v2.pth

  # Test on a specific dataset only:
  python evaluate.py --backbone mobilenet --dataset sos
  python evaluate.py --backbone mobilenet --dataset mklab
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

import config
from src.model import OilSpillModel, OilSpillModelV1
from src.utils import load_checkpoint, dice_score, iou_score
from src.classifier import LookAlikeClassifier
from src.features import extract_features


# ── Metrics helpers ────────────────────────────────────────────────────────────

def compute_m1_metrics(model, test_loader, device):
    """Run Module 1 on the test DataLoader and return mean Dice and IoU."""
    model.eval()
    total_dice, total_iou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="M1 evaluation", leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            if logits.shape != masks.shape:
                logits = F.interpolate(logits, size=masks.shape[-2:],
                                       mode='bilinear', align_corners=False)
            total_dice += dice_score(logits, masks).item()
            total_iou  += iou_score(logits, masks).item()
            n += 1
    if n == 0:
        return 0.0, 0.0
    return total_dice / n, total_iou / n


def compute_m2_metrics(model, img_dir, label_dir, classifier, device, dataset_type="sos"):
    """
    Run Module 1 inference then Module 2 classification on a directory of images.

    For SOS images  → binary ground-truth masks (0/1).
    For MKLab images → class 1 = real oil, class 2 = look-alike (used as negatives).

    Returns:
        dict with keys: tp, fp, tn, fn, precision, recall, f1, accuracy
    """
    valid_exts = ('.png', '.jpg', '.tif', '.tiff')
    image_files = sorted([
        f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)
    ])

    tp, fp, tn, fn = 0, 0, 0, 0
    model.eval()

    for fname in tqdm(image_files, desc=f"M1+M2 ({dataset_type})", leave=False):
        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]

        # ── Load and preprocess image ──
        try:
            pil_img = Image.open(img_path).convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
            )
        except Exception:
            continue

        img_np = np.array(pil_img)
        gray_np = img_np[:, :, 0]

        # ── Module 1 inference ──
        from src.dataset import get_transforms
        dummy_mask = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        aug = get_transforms(train=False)(image=img_np, mask=dummy_mask)
        tensor = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.float32)

        # ── Load ground-truth label ──
        if dataset_type == "sos":
            mask_path = os.path.join(label_dir, fname)
            if not os.path.exists(mask_path):
                continue
            gt = np.array(
                Image.open(mask_path).convert("L").resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST
                )
            )
            gt_oil_mask    = (gt > 127).astype(np.float32)
            gt_look_mask   = None
        else:  # mklab
            label_path = os.path.join(label_dir, base + ".png")
            if not os.path.exists(label_path):
                continue
            gt = np.array(
                Image.open(label_path).convert("L").resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST
                )
            )
            gt_oil_mask  = (gt == 1).astype(np.float32)
            gt_look_mask = (gt == 2).astype(np.float32)

        # ── Module 2: classify each detected patch ──
        if pred_mask.sum() < 50:
            # No significant patch detected — count as TN if no real oil, FN if oil present
            if gt_oil_mask.sum() == 0:
                tn += 1
            else:
                fn += 1
            continue

        features = extract_features(pred_mask, gray_np)
        if not features:
            continue

        predictions = classifier.predict(features)

        # Majority vote: if any patch is oil, image is oil
        any_oil = any(p["is_oil"] for p in predictions)

        gt_has_oil  = gt_oil_mask.sum() >= 50
        gt_has_look = (gt_look_mask.sum() >= 50) if gt_look_mask is not None else False

        if any_oil and gt_has_oil:
            tp += 1
        elif any_oil and not gt_has_oil:
            fp += 1
        elif not any_oil and not gt_has_oil:
            tn += 1
        elif not any_oil and gt_has_oil:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / (total + 1e-9)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "total":     total,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Module 1 + Module 2 pipeline on test set"
    )
    parser.add_argument(
        "--backbone",
        choices=["mobilenet", "efficientnet"],
        default="mobilenet",
        help="Which Module 1 backbone to use (default: mobilenet)"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        help="Path to Module 1 .pth checkpoint"
    )
    parser.add_argument(
        "--classifier",
        default=os.path.join(config.CHECKPOINT_DIR, "module2_classifier.pkl"),
        help="Path to Module 2 .pkl classifier"
    )
    parser.add_argument(
        "--dataset",
        choices=["sos", "mklab", "combined"],
        default="combined",
        help="Which test set to evaluate on (default: combined)"
    )
    parser.add_argument(
        "--skip-m2",
        action="store_true",
        help="Only run Module 1 evaluation (skip Module 2)"
    )
    args = parser.parse_args()

    device = config.DEVICE
    print("\n" + "=" * 65)
    print(f"  OIL SPILL PIPELINE EVALUATION")
    print(f"  Backbone  : {args.backbone.upper()}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset   : {args.dataset.upper()}")
    print(f"  Device    : {device}")
    print("=" * 65)

    # ── Load Module 1 ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading Module 1 ({args.backbone})...")
    if not os.path.exists(args.checkpoint):
        print(f"  ERROR: Checkpoint not found at '{args.checkpoint}'")
        print("  Place your best_model.pth in outputs/checkpoints/ and retry.")
        return

    if args.backbone == "mobilenet":
        model = OilSpillModelV1().to(device)
        print("  → OilSpillModelV1 (MobileNetV3-Large, 256×256)")
    else:
        model = OilSpillModel().to(device)
        print("  → OilSpillModel (EfficientNet-B4, 512×512)")

    load_checkpoint(args.checkpoint, model)
    model.eval()
    print("  → Checkpoint loaded successfully")

    # ── Load Module 2 ──────────────────────────────────────────────────────────
    classifier = None
    if not args.skip_m2:
        print(f"\n[2/3] Loading Module 2 classifier...")
        if not os.path.exists(args.classifier):
            print(f"  ERROR: Classifier not found at '{args.classifier}'")
            print("  Run 'python train_module2.py' first.")
            return
        classifier = LookAlikeClassifier()
        classifier.load(args.classifier)
        print("  → Classifier loaded successfully")

    # ── Module 1 test set evaluation ───────────────────────────────────────────
    print(f"\n[3/3] Running evaluation on test set...")
    from src.dataset import get_loaders

    _, _, test_loader = get_loaders(dataset=args.dataset)

    m1_dice, m1_iou = compute_m1_metrics(model, test_loader, device)

    print("\n" + "─" * 65)
    print(f"  MODULE 1 RESULTS  ({args.backbone.upper()})")
    print("─" * 65)
    print(f"  Test Dice  : {m1_dice:.4f}")
    print(f"  Test IoU   : {m1_iou:.4f}")

    # ── Module 1 + Module 2 combined evaluation ────────────────────────────────
    if not args.skip_m2 and classifier is not None:
        print("\n" + "─" * 65)
        print(f"  MODULE 1 + MODULE 2 COMBINED RESULTS")
        print("─" * 65)

        datasets_to_eval = []
        if args.dataset in ("sos", "combined"):
            datasets_to_eval.append(("sos", config.TEST_IMG_DIR, config.TEST_MASK_DIR))
        if args.dataset in ("mklab", "combined"):
            datasets_to_eval.append(("mklab", config.MKLAB_TEST_IMG_DIR, config.MKLAB_TEST_LABEL_DIR))

        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
        for ds_name, img_dir, label_dir in datasets_to_eval:
            if not os.path.isdir(img_dir):
                print(f"  Skipping {ds_name.upper()} — directory not found: {img_dir}")
                continue
            r = compute_m2_metrics(model, img_dir, label_dir, classifier, device, ds_name)
            print(f"\n  [{ds_name.upper()}] Images evaluated: {r['total']}")
            print(f"    TP={r['tp']}  FP={r['fp']}  TN={r['tn']}  FN={r['fn']}")
            print(f"    Precision : {r['precision']:.4f}")
            print(f"    Recall    : {r['recall']:.4f}")
            print(f"    F1 Score  : {r['f1']:.4f}")
            print(f"    Accuracy  : {r['accuracy']:.4f}")
            all_tp += r["tp"]; all_fp += r["fp"]
            all_tn += r["tn"]; all_fn += r["fn"]

        if len(datasets_to_eval) > 1:
            total = all_tp + all_fp + all_tn + all_fn
            prec  = all_tp / (all_tp + all_fp + 1e-9)
            rec   = all_tp / (all_tp + all_fn + 1e-9)
            f1    = 2 * prec * rec / (prec + rec + 1e-9)
            acc   = (all_tp + all_tn) / (total + 1e-9)
            print(f"\n  [COMBINED OVERALL] Images evaluated: {total}")
            print(f"    Precision : {prec:.4f}")
            print(f"    Recall    : {rec:.4f}")
            print(f"    F1 Score  : {f1:.4f}")
            print(f"    Accuracy  : {acc:.4f}")

    print("\n" + "=" * 65)
    print(f"  Backbone : {args.backbone.upper()}")
    print(f"  M1 IoU   : {m1_iou:.4f}   |   M1 Dice : {m1_dice:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
