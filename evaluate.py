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

def evaluate_single_model(backbone, checkpoint_path, classifier, args, device):
    print("\n" + "=" * 65)
    print(f"  EVALUATING BACKBONE: {backbone.upper()}")
    print("=" * 65)

    print(f"\n[1/2] Loading Module 1 ({backbone})...")
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint not found at '{checkpoint_path}'")
        return None

    if backbone == "mobilenet":
        model = OilSpillModelV1().to(device)
        print("  → OilSpillModelV1 (MobileNetV3-Large, 256×256)")
    else:
        model = OilSpillModel().to(device)
        print("  → OilSpillModel (EfficientNet-B4, 512×512)")

    load_checkpoint(checkpoint_path, model)
    model.eval()
    print("  → Checkpoint loaded successfully")

    print(f"\n[2/2] Running evaluation on test set...")
    from src.dataset import get_loaders
    _, _, test_loader = get_loaders(dataset=args.dataset)

    m1_dice, m1_iou = compute_m1_metrics(model, test_loader, device)
    
    print("\n" + "─" * 65)
    print(f"  MODULE 1 RESULTS  ({backbone.upper()})")
    print("─" * 65)
    print(f"  Test Dice  : {m1_dice:.4f}")
    print(f"  Test IoU   : {m1_iou:.4f}")

    results = {"m1_iou": m1_iou, "m1_dice": m1_dice, "m2_overall": None}

    if not args.skip_m2 and classifier is not None:
        datasets_to_eval = []
        if args.dataset in ("sos", "combined"):
            datasets_to_eval.append(("sos", config.TEST_IMG_DIR, config.TEST_MASK_DIR))
        if args.dataset in ("mklab", "combined"):
            datasets_to_eval.append(("mklab", config.MKLAB_TEST_IMG_DIR, config.MKLAB_TEST_LABEL_DIR))

        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
        for ds_name, img_dir, label_dir in datasets_to_eval:
            if not os.path.isdir(img_dir):
                continue
            r = compute_m2_metrics(model, img_dir, label_dir, classifier, device, ds_name)
            all_tp += r["tp"]; all_fp += r["fp"]
            all_tn += r["tn"]; all_fn += r["fn"]

        total = all_tp + all_fp + all_tn + all_fn
        prec  = all_tp / (all_tp + all_fp + 1e-9)
        rec   = all_tp / (all_tp + all_fn + 1e-9)
        f1    = 2 * prec * rec / (prec + rec + 1e-9)
        acc   = (all_tp + all_tn) / (total + 1e-9)
        
        results["m2_overall"] = {
            "total": total, "tp": all_tp, "fp": all_fp, "tn": all_tn, "fn": all_fn,
            "prec": prec, "rec": rec, "f1": f1, "acc": acc
        }
        
        print("\n" + "─" * 65)
        print(f"  MODULE 1 + MODULE 2 COMBINED RESULTS ({backbone.upper()})")
        print("─" * 65)
        print(f"  Images evaluated: {total}")
        print(f"    TP={all_tp}  FP={all_fp}  TN={all_tn}  FN={all_fn}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    Accuracy  : {acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Module 1 + Module 2 pipeline on test set"
    )
    parser.add_argument(
        "--backbone",
        choices=["mobilenet", "efficientnet", "both"],
        default="both",
        help="Which Module 1 backbone to evaluate (default: both)"
    )
    parser.add_argument(
        "--mobilenet-ckpt",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model_mobilenet.pth"),
        help="Path to MobileNetV3 .pth checkpoint"
    )
    parser.add_argument(
        "--efficientnet-ckpt",
        default=os.path.join(config.CHECKPOINT_DIR, "best_model_efficientnet.pth"),
        help="Path to EfficientNet-B4 .pth checkpoint"
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
    print(f"  Dataset   : {args.dataset.upper()}")
    print(f"  Device    : {device}")
    print("=" * 65)

    classifier = None
    if not args.skip_m2:
        if not os.path.exists(args.classifier):
            print(f"  ERROR: Classifier not found at '{args.classifier}'")
            return
        classifier = LookAlikeClassifier()
        classifier.load(args.classifier)

    configs = []
    if args.backbone in ["mobilenet", "both"]:
        configs.append(("mobilenet", args.mobilenet_ckpt))
    if args.backbone in ["efficientnet", "both"]:
        configs.append(("efficientnet", args.efficientnet_ckpt))

    all_results = {}
    for bb, ckpt in configs:
        res = evaluate_single_model(bb, ckpt, classifier, args, device)
        if res:
            all_results[bb] = res

    # Print final comparison table
    if len(all_results) > 0:
        print("\n\n" + "═" * 70)
        print("  FINAL COMPARISON SUMMARY")
        print("═" * 70)
        print(f"{'Model':<18} | {'M1 IoU':<8} | {'M2 Prec':<8} | {'M2 Recall':<9} | {'Pipeline F1'}")
        print("─" * 70)
        for bb, res in all_results.items():
            m1_iou = f"{res['m1_iou']:.4f}"
            if res["m2_overall"]:
                prec = f"{res['m2_overall']['prec']:.4f}"
                rec = f"{res['m2_overall']['rec']:.4f}"
                f1 = f"{res['m2_overall']['f1']:.4f}"
            else:
                prec, rec, f1 = "N/A", "N/A", "N/A"
            print(f"{bb.upper():<18} | {m1_iou:<8} | {prec:<8} | {rec:<9} | {f1}")
        print("═" * 70 + "\n")

if __name__ == "__main__":
    main()
