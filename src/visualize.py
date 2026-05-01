import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import config
from src.dataset import SOSDataset, get_transforms
from src.model import get_model
from src.utils import load_checkpoint, iou_score, dice_score

def visualize_predictions(num_samples=5):
    device = config.DEVICE
    model = get_model(device)
    load_checkpoint(
        os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        model
    )
    model.eval()

    dataset = SOSDataset(
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        transform=get_transforms(train=False)
    )

    os.makedirs(config.PRED_DIR, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, num_samples * 4.5))
    fig.suptitle(
        "Oil Spill Detection — Module 1 Results\nDeepLabv3+ / MobileNetV2 + scSE Attention",
        fontsize=14, fontweight="bold", y=1.01
    )

    col_titles = ["SAR Input (Sentinel-1)", "Ground Truth Mask", "Predicted Mask"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for row, idx in enumerate(sample_indices):
        image, mask = dataset[idx]

        with torch.no_grad():
            pred_logit = model(image.unsqueeze(0).to(device))
            pred_prob  = torch.sigmoid(pred_logit).squeeze().cpu().numpy()

        pred_binary = (pred_prob > 0.5).astype(np.float32)
        mask_np     = mask.squeeze().numpy()

        # Denormalize image for display
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 0.5 + 0.5).clip(0, 1)
        sar_gray = image_np[:, :, 0]

        # Per-sample metrics
        pred_tensor = pred_logit
        mask_tensor = mask.unsqueeze(0).to(device)
        sample_iou  = iou_score(pred_tensor, mask_tensor).item()
        sample_dice = dice_score(pred_tensor, mask_tensor).item()

        # Column 0 — SAR Input
        axes[row][0].imshow(sar_gray, cmap="gray")
        axes[row][0].set_ylabel(
            f"Sample {idx}\nIoU: {sample_iou:.3f} | Dice: {sample_dice:.3f}",
            fontsize=9, rotation=0, labelpad=80, va="center"
        )

        # Column 1 — Ground Truth
        axes[row][1].imshow(sar_gray, cmap="gray")
        gt_overlay = np.zeros((*mask_np.shape, 4))
        gt_overlay[mask_np == 1] = [1, 0, 0, 0.55]
        axes[row][1].imshow(gt_overlay)

        # Column 2 — Prediction
        axes[row][2].imshow(sar_gray, cmap="gray")
        pred_overlay = np.zeros((*pred_binary.shape, 4))
        pred_overlay[pred_binary == 1] = [0, 0.8, 1, 0.55]
        axes[row][2].imshow(pred_overlay)

        for col in range(3):
            axes[row][col].axis("off")

    # Legend
    gt_patch   = mpatches.Patch(color=(1, 0, 0, 0.7),    label="Ground Truth (Red)")
    pred_patch = mpatches.Patch(color=(0, 0.8, 1, 0.7),  label="Prediction (Cyan)")
    fig.legend(
        handles=[gt_patch, pred_patch],
        loc="lower center", ncol=2,
        fontsize=11, frameon=True,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()
    save_path = os.path.join(config.PRED_DIR, "predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_predictions()